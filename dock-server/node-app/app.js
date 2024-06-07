var mysql = require('mysql2');
var axios = require('axios');
var port = 3100;
var con = mysql.createConnection({
  host: "10.163.234.251",
  user: "root",
  password: "admin",
  database: "data"
});

con.connect(function(err) {
    if (err) throw err;
    console.log("Connected!");
});

var data1;
async function scrape(){
    return await axios.get("http://10.163.234.251:9100/metrics").then(function(response) {
        data1 = response.data;
//        console.log(data1);
    }).catch(error => {
        console.log(error)
    });
}

setInterval(() => {
    scrape();
    setTimeout(() => {
        ts = new Date().toISOString('Asia/Kolkata').replace(/T/, ' ').replace(/\..+/, '');
        var x=data1.split('\n');
        var a;
        for (let i in x){
            // console.log(x[i]);
            if (x[i].includes('node_load1 ')){
                if (x[i].includes('#')) continue;
                var j= x[i].split(' ');
                a = parseFloat(j[1]);
                // console.log("check");
                if (a>1) a=1;
            }}
        let add = [ts, a*100];
        con.query(`INSERT INTO live223(timestamp,cpuV) VALUES(?,?)`, add, function (err, result) {
        if (err) throw err;
        console.log("Result: " + result + ts + a);
        });
    }, 3500)
}, 20000)