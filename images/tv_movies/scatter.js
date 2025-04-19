// http://bl.ocks.org/peterssonjonas/4a0e7cb8d23231243e0e

var version = document.currentScript.getAttribute('tvmovie');
if (version == 'tv'){
  var init_x = "first_air_date",
      filter_level = 10,
      csv_file = "shows.csv",
      xLabel = 'Date of First Airing',
      tipLabel = "Show Title: ";
}
else if (version == 'movie'){
  var init_x = "release_date",
      filter_level = 1000,
      csv_file = "movies.csv",
      xLabel = 'Release Date',
      tipLabel = "Movie Title: ";
}

// plot placement
var margin = { top: 50, right: 190, bottom: 50, left: 50 },
    outerWidth = 800,
    outerHeight = 500,
    width = outerWidth - margin.left - margin.right,
    height = outerHeight - margin.top - margin.bottom;

var parseTime = d3.time.format("%Y-%m-%d").parse;

// field names
var xCat = "first_air_date",
    yCat = "vote_average",
    rCat = "popularity",
    title = "show_name",
    colorCat = "genre_ids";

// function for linear regression
function linearRegression(y,x){

    var lr = {};
    var n = y.length;
    var sum_x = 0;
    var sum_y = 0;
    var sum_xy = 0;
    var sum_xx = 0;
    var sum_yy = 0;

    for (var i = 0; i < y.length; i++) {

        sum_x += x[i];
        sum_y += y[i];
        sum_xy += (x[i]*y[i]);
        sum_xx += (x[i]*x[i]);
        sum_yy += (y[i]*y[i]);
    }

    lr['slope'] = (n * sum_xy - sum_x * sum_y) / (n*sum_xx - sum_x * sum_x);
    lr['intercept'] = (sum_y - lr.slope * sum_x)/n;
    lr['r2'] = Math.pow((n*sum_xy - sum_x*sum_y)/Math.sqrt((n*sum_xx-sum_x*sum_x)*(n*sum_yy-sum_y*sum_y)),2);

    return lr;

};

// make the plot. start by reading in the data
d3.csv(csv_file, function(data) {
  data.forEach(function(d) {
    d.first_air_date = parseTime(d[init_x]);
    d.vote_average = +d.vote_average;
    d.popularity = +d.popularity;
    d.show_name = d.show_name;
    d.vote_count = +d.vote_count;
  });

  // run the linear regression
  var yval = data
      .filter(function(d) { return d.vote_count > filter_level })
      .map(function (d) { return parseFloat(d.vote_average); });
  var xval = data
      .filter(function(d) { return d.vote_count > filter_level })
      .map(function (d) { return d.first_air_date.getTime() / 1000; });
  var lr = linearRegression(yval,xval);
  console.log(lr)
  // define the x scale (horizontal)
  var mindate = new Date(1950,0,1),
      maxdate = new Date(2017,0,31);

  // create functions for understanding x and y data
  var y = d3.scale.linear().range([height, 0]).nice().domain([3, 9]);
  var x = d3.time.scale()
      .domain([mindate, maxdate])
      .range([0, width - 0 * 2]);

  var yMax = d3.max(data, function(d) { return d[yCat]; }),
      yMin = d3.min(data, function(d) { return d[yCat]; }),
      yMin = yMin > 0 ? 0 : yMin;

  // create the x axis variable
  var xAxis = d3.svg.axis()
      .scale(x)
      .orient("bottom")
      .tickSize(-height)
      .tickFormat(d3.time.format("%Y"));

  // create the y axis variable
  var yAxis = d3.svg.axis()
      .scale(y)
      .orient("left")
      .tickSize(-width);

  var color = d3.scale.category20();

  // data labels
  var tip = d3.tip()
      .attr("class", "d3-tip")
      .offset([-10, 0])
      .html(function(d) {
        return tipLabel + d['show_name'] + "<br>" + "Rating: " + d[yCat];
      });

  // control zooming
  var zoomBeh = d3.behavior.zoom()
      .x(x)
      .y(y)
      .scaleExtent([1, 5])
      .on("zoom", zoom);

  // draw area for scatter plot
  var svg = d3.select("#scatter")
    .append("svg")
      .attr("width", outerWidth)
      .attr("height", outerHeight)
    .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
      .call(zoomBeh);

  svg.call(tip);

  svg.append("rect")
      .attr("width", width)
      .attr("height", height);

  // place x axis
  svg.append("g")
      .classed("x axis", true)
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
      .style("font-size","12px");

  // move x axis text down 10px
  svg.selectAll('text')
      .attr("y", 10);

  // place x axis label
  svg.append("g")
      .append("text")
      .classed("label", true)
      .attr("x", width*.5+margin.left)
      .attr("y", height+margin.bottom-10)
      .style("text-anchor", "end")
      .text(xLabel);

  // place y axis
  svg.append("g")
      .classed("y axis", true)
      .call(yAxis)
      .style("font-size","12px");

  // place y axis label
  svg.append("g")
      .append("text")
      .classed("label", true)
      .attr("transform", "rotate(-90)")
      .attr("x", 0-(height/2)+margin.bottom)
      .attr("y", -margin.left+5)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text('Average Score');

  // create object for drawing axis lines
  var objects = svg.append("svg")
      .classed("objects", true)
      .attr("width", width)
      .attr("height", height);

  // draw x axis line
  objects.append("svg:line")
      .classed("axisLine hAxisLine", true)
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", width)
      .attr("y2", 0)
      .attr("transform", "translate(0," + height + ")");

  // draw y axis line
  objects.append("svg:line")
      .classed("axisLine vAxisLine", true)
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", height);

  // draw the points in the scatterplot
  objects.selectAll(".dot")
      .data(data)
      .enter().append("circle")
      .filter(function(d) { return d.vote_count > filter_level })
      .classed("dot", true)
      .attr("r", function (d) { return 6 * Math.sqrt(d[rCat] / Math.PI); })
      .attr("transform", transform)
      .style("fill", function(d) { return color(d[colorCat]); })
      .on("mouseover", tip.show)
      .on("mouseout", tip.hide);

  // create legend
  var legend = svg.selectAll(".legend")
      .data(color.domain())
      .enter().append("g")
      .classed("legend", true)
      .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

  legend.append("circle")
      .attr("r", 3.5)
      .attr("cx", width + 20)
      .attr("fill", color);

  legend.append("text")
      .attr("x", width + 26)
      .attr("dy", ".35em")
      .text(function(d) { return d; });

  // function that controls how zooming in and out plays out.
  function zoom() {
    svg.select(".x.axis").call(xAxis);
    svg.select(".y.axis").call(yAxis);

    svg.selectAll(".dot")
        .attr("transform", transform);
  }

  function transform(d) {
    return "translate(" + x(d[xCat]) + "," + y(d[yCat]) + ")";
  }
});
