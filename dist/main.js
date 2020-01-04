
///////////////////////////////////// Global Variables ////////////////////////////////                                                                                            
                                                                                           

let myGroups = []
let myIncomingClassifier = []                                                                                          
let    classifier = knnClassifier.create();
                                                                                            
//////////////////////////////////////// knn classifier stuff///////////////////////////



myRunFace = async function() {
 // clearInterval(document.myTimer)                                                                                          

  // mobilenetModule = await mobilenet.load();
  net = new faceapi.FaceLandmark68TinyNet()       // made it global                                                                                      
  await net.load('https://hpssjellis.github.io/beginner-tensorflowjs-examples-in-javascript/advanced-keras/face/models/face_landmark_68_tiny_model-weights_manifest.json')                                                                                           
                                                                                            
  console.log('knn-classigfier and face-api loaded')
 // document.myTimer = setInterval(myPredict, 1000)                                                                                           
}


myTextChange = async function(){                                                                                               
   if (parseInt(document.getElementById('myClassNumber').value) < 0 ){
          document.getElementById('myClassNumber').value = 0                                                                                  
   }                         
   if (myGroups[document.getElementById('myClassNumber').value] == undefined){
       document.getElementById('myClassText').value    = ''   // clear the box                                                                                     
     } else {                                                                                         
       document.getElementById('myClassText').value    = myGroups[document.getElementById('myClassNumber').value] 
   }
}                                                                                             

myNewTrain = async function(){   

   let myCanvasElement = document.getElementById('my32x32CanvasA');
  let myCTX = myCanvasElement.getContext('2d');
  myCTX.drawImage(myVideoStream, 0, 0, myCanvasElement.width, myCanvasElement.height);                                                                                           
  const img1 = tf.browser.fromPixels(my32x32CanvasA);                                                                                            
 // const { boxes, scores } = await net.forward(img1) 
  //console.log('boxes and scores loaded')  
   
 const wow = await net.detectLandmarks(img1)                                                                                           
 //console.log('wow')                                                                                                                    
 //console.log(wow)                                                                                                                     
 //console.log('wow._faceLandmarks')                                                                                                      
 //console.log(wow._faceLandmarks)                                                                                                  
 //console.log('wow._positions')                                                                                                      
 //console.log(wow._positions)  
 
                                                                                            
 newArray = []                                                                                           

                                                                                            
   
 for (let j=0;  j < wow._positions.length; j++ ){  
    newArray[j] = []                                                                                        
    newArray[j][0] = wow._positions[j].x                                                                
    newArray[j][1] = wow._positions[j].y  
  }                                                                                          
   
                                                                                                                                                                                   
                                                                                            
  classifier.addExample(tf.tensor2d(newArray, shape=[68,2], 'float32'), parseInt(document.getElementById('myClassNumber').value));                                                              
  myGroups[document.getElementById('myClassNumber').value] = document.getElementById('myClassText').value.replace(' ','_')

  const myAddCount = classifier.getClassExampleCount()[document.getElementById('myClassNumber').value]
  const myGroupNumber =  document.getElementById('myClassNumber').value                                                                                         
  document.getElementById('myDivLoss').innerHTML = myAddCount + ' times, Example Added: to group #: '+ myGroupNumber 
                                                                                    
                                                                                            
                                                                                                                                                                                  
                                                                                            
                                                                                            
                                                                                            
}
                                                                                            
                                                                                            
                                                                                            
                                                                                            
myPredict = async function(){
  //just to clear the other canvas
  let myCanvasElement = document.getElementById('my32x32CanvasA');
  let myCTX = myCanvasElement.getContext('2d');                                                                                            
  myCTX.clearRect(0, 0, myCanvasElement.width, myCanvasElement.height);                                                                                          
                                                                                            
 if (classifier.getNumClasses() >= 1){
                                                                                                                                                                                    
  const img2 = tf.browser.fromPixels(myVideo);                                                                                            
  const wow2 = await net.detectLandmarks(img2)                                                                                                                                                                                       
  newArray2 = []                                                                                           

                                                                                            
   
 for (let j=0;  j < wow2._positions.length; j++ ){  
    newArray2[j] = []                                                                                        
    newArray2[j][0] = wow2._positions[j].x                                                                
    newArray2[j][1] = wow2._positions[j].y  
  }                                                                                          
   
                                                                                                                                                                                   

  const result = await classifier.predictClass(tf.tensor2d(newArray2, shape=[68,2], 'float32'), 3);   // number of groups
   
  document.getElementById('myDivTest').innerHTML = 'Group: ' + result.classIndex + ', '+ 
      Math.round(result.confidences[result.classIndex]*100)+ '% ' + myGroups[result.classIndex]+ '<br>' 
 } else {(console.log('Need to train groups'))}
}
                                                                                            

/////////////////////////////////////////// NEW SAVING and LOADING functions ///////////////////////////////////////////                                                                                      
                                                                                            
                                                                                            
myDefineClassifierModel = async function(myPassedClassifier){

   let myLayerList = []
    myLayerList[0] = []    // for the input layer name as a string
    myLayerList[1] = []    // for the input layer
    myLayerList[2] = []    // for the concatenate layer name as a string
    myLayerList[3] = []    // for the concatenate layer
                                                                                                                                                                                       
   let myMaxClasses = myPassedClassifier.getNumClasses()                                                                                      
   console.log('myPassedClassifier.getNumClasses()')
   console.log(myMaxClasses)                                                                                      
                                                                                            
   for (let myClassifierLoop = 0; myClassifierLoop < myMaxClasses; myClassifierLoop++ ){      // need number of classifiers                                        
                                                                                        
     console.log(myPassedClassifier.getClassifierDataset()[myClassifierLoop])                                                                                                                                                           
     console.log('shape first layer =')                                                                                                                                                           
     console.log(myPassedClassifier.getClassifierDataset()[myClassifierLoop].shape[0])
                                                                                            
     myLayerList[0][myClassifierLoop] = 'myInput'  + myClassifierLoop                  // input name as a string
     console.log('define input for'+myClassifierLoop)                                                                                       
     myLayerList[1][myClassifierLoop] = tf.input({shape: myPassedClassifier.getClassifierDataset()[myClassifierLoop].shape[0], name: myLayerList[1][myClassifierLoop]});      // Define input layer
     console.log('define dense for: '+myClassifierLoop)
     myLayerList[2][myClassifierLoop] = 'myInput'+myClassifierLoop+'Dense1'    // concatenate as a string                                                                                  
     myLayerList[3][myClassifierLoop] = tf.layers.dense({units: 136, name: myGroups[myClassifierLoop]}).apply(myLayerList[1][myClassifierLoop]);             //Define concatenate layer                                                                          
                                                                                          
   }
                                                                                            

  // what the layers used to look like before the loop                                                                                            
  //const myInput2 = tf.input({shape: [1], name: 'myInput2'});
  //const myInput2Dense1 = tf.layers.dense({units: 20, name: 'myInput2Dense1'}).apply(myInput2);

                                                                                            
  console.log('Concatenate Paths')                                                                                                                                                                                          
  const myConcatenate1 = tf.layers.concatenate({axis : 1, name: 'myConcatenate1'}).apply(myLayerList[3]);    // send the entire list of dense                                                                                           
  const myConcatenate1Dense4 = tf.layers.dense({units: 1, name: 'myConcatenate1Dense4'}).apply(myConcatenate1)                                                                                              

  console.log('Define Model')                                                                                                                                                                                       
  const myClassifierModel = tf.model({inputs: myLayerList[1], outputs: myConcatenate1Dense4});    // This would be a global model. With list of inputs as an array                                                                                                                                                                                         
  myClassifierModel.summary()
  console.log('myClassifierModel.layers[myMaxClasses]')     
  console.log(myClassifierModel.layers[myMaxClasses])
  myPassedClassifier.getClassifierDataset()[0].print(true)                                                                                            
                                                                                            
  for (let myClassifierLoop = 0; myClassifierLoop < myMaxClasses; myClassifierLoop++ ){   // since the first layers are inputs must add maxClasses   
    const myInWeight = await myPassedClassifier.getClassifierDataset()[myClassifierLoop]                                                                                        
    myClassifierModel.layers[myClassifierLoop + myMaxClasses].setWeights([myInWeight, tf.ones([136])]);       //model.layers[0].setWeights([tf.ones([10, 2]), tf.ones([2])]);                                                                                        

 }                                                                                           
                                                                                            
return  myClassifierModel                                                                                        
}                                                                                             
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                         
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
///////////////////////////////////////////////////////////////////////////////                                                                                             
myClassifierSave  = async function(){
  try{
    const myClassifierModel2 = await myDefineClassifierModel(classifier)         // pass global classifier                                                                                                                                                                                                                                                                      
    myClassifierModel2.save('file:///./myClassifierModel01')  
    myClassifierModel2.summary(null,null,x => {document.getElementById('myDivSummary').innerHTML += x + '<br>'});  
  }catch(error){
      console.log(error);
  }                                                                                                                                                                                     
                                                                                                                                                                                                                                                                             
}
                                                                                            

                                                                                            
                                                                                            
/////////////////////////////////////////////                                                                                             
mySetClassiferModelWeights  = async function(){
                                                                                            
                                                                                            
}                                                                                              
 
                                                                                               
                                                                                            
//////////////////////////////////////////////////////////////////////////////
myClassifierLoad  = async function(){
  // note global variable called        myIncomingClassifier        
                                                                                            
  const myLoadedModel  = await	tf.loadLayersModel(document.getElementById('myInFile').value)                                                                                          
  console.log('myLoadedModel.layers.length')   
  console.log(myLoadedModel.layers.length) 
                                          
                                          
 // console.log('myLoadedModel.layers[0].batchInputShape[1]')   
 // console.log(myLoadedModel.layers[0].batchInputShape[1] )       
                                                                                  
 const myMaxLayers = myLoadedModel.layers.length
 const myDenseEnd =  myMaxLayers - 2                                                                                          
 const myDenseStart = myDenseEnd/2         // assume 0 = first layer: if 6 layers 0-1 input, 2-3 dense, 4 concatenate, 5 dense output                                                                                  
                                                                                           
  for (let myWeightLoop = myDenseStart; myWeightLoop < myDenseEnd; myWeightLoop++ ){      // need number of classifiers                                        
                                             
     // console.log('myLoadedModel.layers['+myWeightLoop+']')   
     // console.log(myLoadedModel.layers[myWeightLoop])          
     console.log('myLoadedModel.layers['+myWeightLoop+'].getWeights()[0].print(true)') 
    // myLoadedModel.layers[myWeightLoop].getWeights()[0].print(true)                                                                                        
     myIncomingClassifier[myWeightLoop - myDenseStart] =  myLoadedModel.layers[myWeightLoop].getWeights()[0] 
     myGroups[myWeightLoop - myDenseStart] =  myLoadedModel.layers[myWeightLoop].name     // hopefully the name is the group name                                                                                     
  }
 console.log('Printing all the incoming classifiers')
 for (x=0;  x < myIncomingClassifier.length ; x++){
   myIncomingClassifier[x].print(true)                                                                                          
 }                                                                                           
 console.log('Activating Classifier')   
 
 classifier.dispose() // clear old classifier 
 classifier.setClassifierDataset(myIncomingClassifier)  
 console.log('Classifier loaded')                                                                                              
                                                                                            
}                                                                                             
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
           
///////////////////////////////////// End KNN- Classifier stuff /////////////////////////////////////                                                                                                
///////////////////////////////////// webcam stuff /////////////////////////////////////                                                                                             

                                                                                            
var myVideoStream = document.getElementById('myVideo')     // make it a global variable
var myStoredInterval = 0

                                                                                            
                                                                                            
stopVideo = async function() {  
 clearInterval(myStoredInterval)   // god idea to stop the auto snapshot taking                                                                                         
 myVideoStream.srcObject.getTracks().forEach(track => track.stop())  
}
  
  
                                                                                            
                                                                                            
                                                                                            
getVideo = async function() {  
 const myCamera = await document.getElementById('myCheck').value
                                                                                    
 navigator.getMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
 navigator.getMedia({video: { facingMode: myCamera }, audio: false},
                    
   function(stream) {
     myVideoStream.srcObject = stream   
     myVideoStream.play();
 }, 
                    
  function(error) {
    alert('webcam not working');
 });
}
  
                                                                                            
                                                                                            


                                                                                            
takeAuto = async function(){                                                                                           
   myStoredInterval = setInterval(async function(){                                                                                         
      await myPredict()                                                                                   
  }, document.getElementById('myInterval').value);        
}
 
  
                                                                                            
                                                                                            
myStopAuto  = async function(){                                                                                             
   clearInterval(myStoredInterval)    
}                                                                                            
 
                                                                                            
                                                                                            
                                                                                            
///////////////////////////////////////////// Done Webcam functions ////////////////////////////////////////                                                                                               
                                                                                            
myLoadUrl = async function(){
 //alert('The test function will need to be changed if other models are loaded')                                                                                             
 document.getElementById('myDivTest').innerHTML = 'Expect major code changes if you load a different model than what is expected'  
 const myFileName = document.getElementById('myInFile').value
 if (myFileName != null){  
   model = await tf.loadLayersModel(myFileName);     // should make the model a global variable
   document.getElementById('myDivSummary').innerHTML = ''      
   await model.summary(null,null,x => {document.getElementById('myDivSummary').innerHTML += x + '<br>'});
  // await myPredict()
 }                                                                           
}                                                                                             
                                                                                            

                                                                                            
                                                                                 
                                                                                            
                                                                                            
/////////////////////////////////// END ALL FUNCTIONS ///////////////////////////////////////                                                                                             
                                                                                            
                                                                                            
                                                                                            
//////////////////////////////////   WEIRD STYLE TAG THAT IS ACTUALLY A DYNAMIC SCRIPT TAG ///////////////////                                                                                          
                                                                                            
                                                                                            
