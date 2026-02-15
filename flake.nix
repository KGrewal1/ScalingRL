{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    homebase = {
      url = "github:KGrewal1/homebase";
      flake = false;
    };
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, homebase, pyproject-nix, uv2nix, pyproject-build-systems }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs { inherit system; };
      pkgsUnfree = import nixpkgs {
        inherit system;
        config = { allowUnfree = true; cudaSupport = true; };
      };
      lib = pkgs.lib;

      # Import homebase packages and cuda modules
      base = import "${homebase}/packages/packages.nix" { inherit pkgs lib; };
      cuda = import "${homebase}/packages/cuda.nix" {
        cudaPackages = pkgsUnfree.cudaPackages_13;
        inherit lib;
      };

      # uv2nix workspace
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      python = pkgs.python312;

      pythonSet =
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (lib.composeManyExtensions [
            pyproject-build-systems.overlays.default
            overlay
          ]);

      venv = pythonSet.mkVirtualEnv "scalingrl-env" workspace.deps.default;

      # Docker helpers (reused from homebase patterns)
      inherit (pkgs.dockerTools) usrBinEnv binSh caCertificates;

      userNss = pkgs.runCommand "user-nss" {} ''
        mkdir -p $out/etc
        cat > $out/etc/passwd <<'EOF'
root:x:0:0:root:/root:/bin/sh
nobody:x:65534:65534:nobody:/nonexistent:/bin/sh
dev:x:1000:1000:dev:/home/dev:${pkgs.fish}/bin/fish
EOF
        cat > $out/etc/group <<'EOF'
root:x:0:
wheel:x:1:dev
nobody:x:65534:
dev:x:1000:
EOF
        cat > $out/etc/nsswitch.conf <<'EOF'
hosts: files dns
EOF
      '';

      sudoSetup = pkgs.runCommand "sudo-setup" {} ''
        mkdir -p $out/etc/sudoers.d $out/etc/pam.d
        echo "%wheel ALL=(ALL) NOPASSWD: ALL" > $out/etc/sudoers.d/wheel
        chmod 440 $out/etc/sudoers.d/wheel
        cat > $out/etc/sudoers <<'EOF'
root ALL=(ALL) ALL
@includedir /etc/sudoers.d
EOF
        chmod 440 $out/etc/sudoers
        cat > $out/etc/pam.d/sudo <<'EOF'
auth       sufficient pam_permit.so
account    sufficient pam_permit.so
session    sufficient pam_permit.so
EOF
      '';

      dockerEnvPaths = [ usrBinEnv binSh caCertificates userNss sudoSetup pkgs.sudo ];

      configFileLinks = lib.concatStringsSep "\n" (lib.mapAttrsToList (target: cfg: ''
        mkdir -p /home/dev/.config/${builtins.dirOf target}
        ln -s ${cfg.source} /home/dev/.config/${target}
      '') base.configFiles);

      homeDirSetup = ''
        mkdir -p /home/dev/.cache /home/dev/.local/share /home/dev/.local/state
        mkdir -p /tmp
        chmod 1777 /tmp
        ${configFileLinks}
        chown -R 1000:1000 /home/dev
      '';

      entrypoint = pkgs.writeShellScript "scalingrl-entry" ''
        cd /home/dev
        exec fish "$@"
      '';
    in
    {
      packages.${system}.docker = pkgs.dockerTools.buildLayeredImage {
        name = "scalingrl-cuda";
        tag = "latest";
        contents = base.packages ++ cuda.packages ++ [ venv ] ++ dockerEnvPaths;
        enableFakechroot = true;
        fakeRootCommands = homeDirSetup;
        config = {
          Cmd = [ "${entrypoint}" ];
          User = "dev";
          Env = [
            "HOME=/home/dev"
            "USER=dev"
            "UV_PYTHON_PREFERENCE=system"
            "UV_PYTHON=${base.env.UV_PYTHON}"
            "LD_LIBRARY_PATH=${base.env.LD_LIBRARY_PATH}:${cuda.env.LD_LIBRARY_PATH}"
            "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            "CUDA_HOME=${cuda.env.CUDA_HOME}"
            "NVIDIA_VISIBLE_DEVICES=all"
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
          ];
        };
      };
    };
}
