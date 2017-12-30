//NodeJS
var express = require('express');
var app = express();
var http = require('http').Server(app);
var io = require('socket.io')(http);

var fs = require('fs');
var path = require('path');
var yaml = require('js-yaml');
var chokidar = require('chokidar');
var format = require('string-format');
format.extend(String.prototype);

var config = require('./config.json');
var port = config.port;
var projectDir = config.projectDir;

//open project's config.yml file in order to find the path for the loss image file
var projectConfig = yaml.safeLoad(fs.readFileSync(projectDir + '/config.yml'));
var modelsDir = projectConfig['dir_name']['models'];
var fileType = projectConfig['charRNNmodel']['file_type'];
var region = projectConfig['region'];

var lossImageFile = path.join(projectDir, modelsDir, '{}_{}_loss.png'.format(fileType, region));
console.log('Watching loss file: ' + lossImageFile);

app.use(express.static(__dirname + '/public/'));

http.listen(port, function() {
    console.log('listening on *:{}'.format(port));
});

io.on('connection', function(socket) {

    chokidar.watch(lossImageFile, {awaitWriteFinish: {stabilityThreshold: 250}}).on('all', function(ev, targetFile) {
        if (ev == 'unlink') return;

        fs.readFile(lossImageFile, function(err, data) {
            if (err) throw err;
            socket.emit('imageData', "data:image/png;base64,"+ data.toString("base64"));
        });

    });

});

