1. Try to centralize the code paths - it is easier to keep definitions in sync if they are defined once
2. Do not import unnecessary libraries and outside of testing code imports should be done at the top of the file
3. Make sure to keep the code clean - do not keep dead APIs, or unneeded code out of concerns of API compatibility - this repo is not a dependency of anything else
4. Keep the readme to the point on how to run the project - do not add fluff
5. This project is managed with modern python using uv - recommend using `uv sync` etc on how to install and other modern idioms
