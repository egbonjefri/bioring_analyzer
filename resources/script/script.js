//Declare variables
var model = undefined;
const spin = document.getElementsByClassName('spinner-border')[0];
const grow = document.getElementsByClassName('spinner-grow')[0];
const displayDiv = document.getElementById('display-div');
const displayImage = document.getElementById('display-image');
const predictionsDiv = document.getElementById('predictionsDiv')
const newStatus = document.getElementById('newStatus');
const myBtn = document.getElementsByClassName('myBtn')[0];
let imageFeatures;
let data = []
const displayImg = document.getElementById('displayImg')
let imageDataInput = []          
const imageInput = document.querySelectorAll('#image-input');
const imageListener = document.getElementById('image-input')
const input_height = 224;
const input_width = 224
let array1 = [];
let circularityIndex = 50
async function loadModel(){
    const URL = 'https://raw.githubusercontent.com/egbonjefri/feature_detector/main/my-model.json';
    model = await tf.loadLayersModel(URL);
    model.summary();
    spin.classList.add('invisible');
    displayDiv.classList.remove('invisible')
    newStatus.innerText = `Model loaded successfully! 
    If uploading batch images for lumen area reduction calculation, please upload them in order...`;  
  }

  loadModel();
  imageListener.addEventListener("change", ()=>{
        if(array1.length > 0){
        let text = document.querySelectorAll('#predictions');
        var svgElement = document.getElementById('svgElement')
        svgElement.parentNode.removeChild(svgElement)
        const canvas = document.querySelectorAll('canvas');
        canvas.forEach((item)=>item.remove());
        text.forEach((item)=>item.parentNode.removeChild(item));
        imageDataInput = []
        array1 = []
    }
    grow.classList.remove('invisible');
    myBtn.classList.add('invisible')
    predict()
  })
myBtn.addEventListener('click', compactionGraph)
  function imageFileToImageElement(imageFile){
    return new Promise((resolve,reject)=>{
        const imageElement = new Image(640,360);
        imageElement.src = URL.createObjectURL(imageFile);
        imageElement.onload = () => {
            URL.revokeObjectURL(imageElement.src);
             imageFeatures = tf.tidy(function() {
              
              imageFrameAsTensor = tf.browser.fromPixels(imageElement);

             let resizedTensorFrame = tf.image.resizeBilinear(imageFrameAsTensor, [input_height, input_width], true);
              let normalizedTensorFrame = resizedTensorFrame.div(255);
        
              return normalizedTensorFrame;
              })
            resolve(imageFeatures);
            
        }
        imageElement.onerror = (error) =>{
            URL.revokeObjectURL(imageElement.src);
            reject(error)
        }
    })
}

  function predict (){

    
    for(let i = 0; i < imageInput[0].files.length; i++){
      async function f1(){
      const x = await imageFileToImageElement(imageInput[0].files[i]);
      return x
    }
    
   imageDataInput.push(f1())
  }
  let bArray = ['bioring', 'not bioring']
 
  Promise.all(imageDataInput).then((values)=>{
    values.forEach((item, index)=>{
   
     async function f1(){
        let output = `canvasOutput${index}`
        let input = `canvas${index}`;
        const canvas = document.createElement('canvas');
        canvas.setAttribute('id', input);
        canvas.style = `  position: absolute;
        left: ${index*14}rem;`
        displayDiv.appendChild(canvas);
        const canvasOutput = document.createElement('canvas');
        canvasOutput.setAttribute('id', output);
        canvasOutput.style = `  position: absolute;
        left: ${index*14}rem;
        opacity: 0.5;`
        displayDiv.appendChild(canvasOutput);
        const canvasImage = await tf.browser.toPixels(item, canvas);


      let expandedDims = item.expandDims()
      let prediction = await model.predict(expandedDims);
      let squeeze = prediction.squeeze()
      const values = squeeze.dataSync();
      let cArray = Array.from(values);
      let predict = bArray[cArray.indexOf(Math.max(...cArray))];
      let confidence = Math.max(...cArray).toFixed(2)*100;
      let newDiv = document.createElement('p');
      newDiv.setAttribute('id', 'predictions')
      if(predict === 'bioring' && confidence > 95){
      drawPolygons(input,output,canvas,index);
      grow.classList.add('invisible');
      myBtn.classList.remove('invisible')
      }
      else{
        newDiv.innerText = `Sorry no bioring detected in the higlighted image(s), please try again with another image...`
        predictionsDiv.appendChild(newDiv);
        canvas.style.border = 'solid 3px red';
        grow.classList.add('invisible');
        data.push({x: index, y: null})

      }
      prediction.dispose();
      item.dispose();
      expandedDims.dispose();
      squeeze.dispose();
      
    }
    f1();

    })
      
   
  })
  array1.push(1);
 
  }

   function drawPolygons(input,output,canvas,index) {

   let array2 = []

  let src = cv.imread(input);

  let dst = cv.Mat.zeros(225, 225, cv.CV_8UC3);
  cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
  cv.threshold(src, src, 30, 150, cv.THRESH_BINARY);
  let contours = new cv.MatVector();
  let hierachy = new cv.Mat();
  cv.findContours(src, contours, hierachy, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS);
  for(let i = 0; i < contours.size(); i++){
    let color = new cv.Scalar(Math.round(Math.random()*255), Math.round(Math.random()*255),Math.round(Math.random()*255));
    let contour = contours.get(i);
    let size = cv.contourArea(contour, true);
    let perimeter = cv.arcLength(contour, true);
    let diff = (2*Math.sqrt(Math.PI*size))
    
    if(size > 500 && (perimeter-diff) <= circularityIndex){
      cv.drawContours(dst, contours, i , color, 2, cv.LINE_4, hierachy, 1);
      array2.push(1);
      data.push({x: index, y: size})
    }

  }
 if(array2.length !== 1){
    let newDiv = document.createElement('p');
    newDiv.setAttribute('id', 'predictions');
    newDiv.innerText = `Can't detect lumen area in one or more images...`
    predictionsDiv.appendChild(newDiv);
    canvas.style.border = 'solid 3px red';
    grow.classList.add('invisible');
    data.push({x: index, y: null})

 }
  cv.imshow(output, dst);
  
 src.delete();
 dst.delete();
 hierachy.delete();
 contours.delete()

  }
  
function compactionGraph(){
  let found = data.find((item)=>{return item.y !== null})
  const baseline = found.y;
  for(let i = 0; i < data.length; i++){
    if(data[i].y===null){
      data.splice(i,1);
    }
    let diff = 1-((baseline - data[i].y)/baseline);
    data[i].y = diff
  }
  var margin = {top: 160, right: 20, bottom: 100, left: 140},
      width = (300*data.length) - margin.left - margin.right,
      height = 500 - margin.top - margin.bottom;

  //set the ranges
  let scaleX = d3.scaleBand()
                .range([0,width])
                .domain(data.map(function(d){return d.x}))
                .padding(0)
                
  ;
  let scaleY = d3.scaleLinear().range([height, 0]);

  var svg = d3.select('body').append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .attr('id', 'svgElement')
            .append('g')
            .attr('transform', 'translate('+margin.left+','+margin.top+')')

  scaleY.domain([0,1]);
  var svgElement = document.getElementById('svgElement')
  svgElement.scrollIntoView({behavior: "smooth", block: "end", inline: "nearest"});

  const tooltip = d3.select('body').append('div').attr('class', 'tooltip-style').style('opacity',0)

  svg.selectAll('circle')
      .data(data)
      .enter().append('circle')
      .attr('class', 'dot')
      .attr('r',5)
      .attr('cx',function(d){return (scaleX(d.x))+scaleX.bandwidth()-(10*data.length)})
      .attr('cy', function(d){return scaleY(d.y)})
      .attr('fill', '#69b3a2')
      .on('mousemove', function(d,event){
        d3.select(this).attr('r',10).style('fill','gray')
        tooltip.style('opacity',1)
        tooltip.html(`Day ${event.x}: <br>
        Percent reduction:
        ${(event.y*100).toFixed(2)}%`)
        tooltip.style('left',(d.pageX)+'px')
        tooltip.style('top',(d.pageY)+'px')
      })
      .on('mouseleave',function(){
        tooltip.style('opacity',0)
        d3.select(this).attr('r',5)
      })
  svg.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "#69b3a2")
      .attr("stroke-width", 1.5)
      .attr("d", d3.line()
        .x(function(d){return Number(scaleX(d.x))+scaleX.bandwidth()-(10*data.length)})
        .y(function(d){return scaleY(d.y)})
        )
  svg.append('g')
    .attr('transform', 'translate(100,'+(height)+')')
    .call(d3.axisBottom(scaleX).ticks(data.length))
   
  svg.append('g')
    .attr('transform', 'translate(100,'+(-5)+')')
    .call(d3.axisLeft(scaleY).tickFormat(d3.format('.0%')))

  svg.append('text')
   .attr('transform', 'translate(60'+' ,'+(100)+')rotate(-90)')
   .style('text-anchor', 'middle')
   .text('Percent Reduction')
   .style('font',"16px sans-serif")
   .attr('fill', 'gray')

   svg.append('text')
   .attr('transform', 'translate('+(width)+' ,'+(height+30)+')')
   .style('text-anchor', 'middle')
   .text('Time')
   .style('font',"16px sans-serif")
   .attr('fill', 'gray')

   //create a table
   var table = d3.select('body').append('table')
                                .attr('class', 'myTable')

   ;

   var thead  = table.append('thead');
   thead.append('tr')
        .selectAll('th')
        .data(['Time', 'Percent Reduction'])
        .enter()
        .append('th')
        .text(function(d) { return d; })
        .style("border", "1px white solid")
        .style("padding", "5px")
        
        .style("background-color", "lightgray")
        .style("font-weight", "bold")
        .style("text-transform", "uppercase");
  var tbody = table.append('tbody')
                    
  var rows = tbody.selectAll('tr')
                  .data(data)
                  .enter()
                  .append('tr')

  var cells = rows.selectAll('td')
                  .data(function(row){
                    return Object.values(row)
                  })
                  .enter()
                  .append('td')
                  .style("border", "1px white solid")
                  .style("padding", "5px")
                  
                  .on("mouseover", function(){
                  d3.select(this).style("background-color", "powderblue");
                })
                  .on("mouseout", function(){
                  d3.select(this).style("background-color", "black");
                })
                  .text(function(d){return d;})
                  .style("font-size", "12px");
          






 }