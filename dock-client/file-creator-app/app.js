var express = require("express");
var axios = require('axios');
var app = express();
const fs = require('fs');

var port = 3200;
var data1;

// function sleep (time) {
//     return new Promise((resolve) => setTimeout(resolve, time));
// }

async function scrape(){
    return await axios.get("http://10.163.234.251:9100/metrics").then(function(response) {
        data1 = String(response.data).replace(/\r?\n/g, "\n");
//        console.log(data1);
    }).catch(error => {
        console.log(error)
    });
}

scrape();

setInterval(() => {
    scrape();
    console.log("in")
    setTimeout(() => {
	console.log(data1);
        fs.writeFile('./data.txt', data1, err => {
            if (err) {
                console.error(err);
            }});
    }, 4500)
}, 40000)

app.get("/download-metrics-file", (req, res) => {
    const file = './data.txt';
    res.download(file, function (err) {
        if (err) {
          console.log(err)
        }
    })
});

app.listen(port, () => {
    console.log("Server listening on port " + port);
});

