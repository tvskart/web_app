express = require('express');
var app = express();
var PythonShell = require('python-shell');

var bodyParser = require('body-parser')
app.use( bodyParser.json() );       // to support JSON-encoded bodies
app.use(bodyParser.urlencoded({     // to support URL-encoded bodies
  extended: true
})); 

//todo: add views, static, middleware parsers, etc
app.use(express.static(__dirname + '/public'));

app.get('/', (req, res) => {
    res.render('index');
})

//TODO: GRADER TO ENTER THEIR PYTHON PATH HERE!!!!!!!!!!!!!!!!
//IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
const pythonPath = 'python'; //'/usr/bin/python3',

// perceptron, neaural net, SVM
app.post('/knn', (req, res) => {
  // console.log(req.body.img_data_url)
  var img_data_url = [req.body.img_data_url];
  var options = {
    pythonPath: pythonPath,
    args: img_data_url
  };

  PythonShell.run('./ai/KNN_image_web.py', options, function (err, results) {
    if (err) {
      console.log(err);
    } else {
      // results is an array consisting of messages collected during execution
      console.log('results:'+ results[results.length-1]);
      res.json({
        result: results[results.length-1]  
      });
    }
  });
  // res.json({
  //   result: '2'
  // });
})

app.post('/nn', (req, res) => {
  // console.log(req.body.img_data_url)
  var img_data_url = [req.body.img_data_url];
  var options = {
    pythonPath: pythonPath,
    args: img_data_url
  };

  PythonShell.run('./ai/test.py', options, function (err, results) {
    if (err) {
      console.log(err);
    } else {
      // results is an array consisting of messages collected during execution
      console.log('results:'+ results[results.length-1]);
      res.json({
        result: results[results.length-1]  
      });
    }
  });
});

app.listen(3000, 'localhost', () => {
  console.log('App listening on http://localhost:3000');
})
