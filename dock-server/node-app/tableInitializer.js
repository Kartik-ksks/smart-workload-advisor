var mysql = require('mysql2');
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

con.query(`create table live223(timestamp DATETIME, cpuV FLOAT);`, function (err, result) {
    if (err) console.log(err);
    console.log("Result: " + result);
});

con.query(`create table next10(timestamp TEXT, cpuV DOUBLE);`, function (err, result) {
    if (err) console.log(err);
    console.log("Result: " + result);
});