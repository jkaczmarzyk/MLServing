# Machine Learning Model Serving Template 

## How to run this template

1. Make sure you have Docker installed
2. Build the image with: 
```
chmod +x ./build_image.sh
./build_image.sh
```
3. Start the container with `./start_container.sh`
4. In a new terminal window, run `./test.sh`

## How to run this template with your own model

1. Save model file(s) to the folder named app
2. Add model specific code to the file model_function.py and model_handler.py
3. Specify further requirements in requirements.txt 
4. Adjust config.yml file with appropriate keys and values