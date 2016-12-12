# web_app
AI Project -  OCR classification

This is a nodejs app that runs a local web server hosting a webpage that user interacts with. Draws a number on the canvas and web app executes the AI algorithms to classify the images and show the results. We show results of our own Neaural Net vs. results of SKLearn's SVM implementation..

How to run?

1) You need node installed.
brew install node
npm install (installs all node packages needed and specified in package.json)

2) You need to specify your python path (python v3 preferably) in the code (index.js)

//TODO: GRADER TO ENTER THEIR PYTHON PATH HERE!!!!!!!!!!!!!!!!
//IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
const pythonPath = 'python'; <<<--- change this here

3) You need many python packages installed that are used in our algorithms

4) Run the server and test the url.
npm start (runs the server)
http://localhost:3000/ (hit this url)

5) Refer to gifs as examples of how to use the app..
