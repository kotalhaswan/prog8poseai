import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const enableWebcamButton = document.getElementById("webcamButton");
const saveData = document.getElementById("saveLocal");

const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

let poseArray = [];
let currentPose = {
    label: "arm-up",
    data: []
}

let poseLandmarker = undefined;
let webcamRunning = false;
let lastVideoTime = -1;
let tickUpdate = false;
let tick = setInterval(tickTimer, 1000);
function tickTimer(){
    tickUpdate =!tickUpdate;
    console.log(tickUpdate)
}
const videoWidth = "480px"
const videoHeight = "270px"
document.getElementById("saveLocal").addEventListener("click", savePoseData);
document.getElementById("poseAdd").addEventListener("click", addNewPose);
document.getElementById("poseLoad").addEventListener("click", loadPosesLocal);
document.getElementById("poseDownload").addEventListener("click", exportJsonData)
document.getElementById("poseRecord").addEventListener("click", setRecording);

let recording = false;
let recordSampleSize = 100;


// ********************************************************************
// if webcam access, load landmarker and enable webcam button
// ********************************************************************
function startApp() {
    const hasGetUserMedia = () => { var _a; return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia); };
    if (hasGetUserMedia()) {
        createPoseLandmarker();
    } else {
        console.warn("getUserMedia() is not supported by your browser");
    }
}
function setRecording(){
    recording = true
}
function savePoseData(){
    localStorage.setItem('poseArray', JSON.stringify(poseArray));
    console.log(JSON.stringify(poseArray))
    fillList();
}
function exportJsonData(){
    let json = JSON.stringify(poseArray);
    download(json, `pose_export.json`, `application/json`);
}

function download(data, filename, type){
    var file = new Blob([data], {type: type});
    if (window.navigator.msSaveOrOpenBlob){
        window.navigator.msSaveOrOpenBlob(file, filename);
    }
    else{
        var a = document.createElement("a"),
            url = URL.createObjectURL(file);
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(function (){
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 0);
    }
}
function loadPosesLocal(){
    poseArray = [];
    let savedPoseJson = localStorage.getItem('poseArray');
    if (savedPoseJson){
        let savedPose = JSON.parse(savedPoseJson);
        for (let i = 0; i < savedPose.length; ++i){
            let newPose = { label: savedPose[i].label, data:savedPose[i].data}
            poseArray.push(newPose);

        }
        fillList();
    }
}

function recordPose(arr){
    if(currentPose.data.length<recordSampleSize)
    {
        currentPose.data.push(arr)
    }
    else
    {
        console.log("FINISHED " + currentPose.label)
        recording = false;
    }
}

// ********************************************************************
// create mediapipe
// ********************************************************************
const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 2
    });
    enableWebcamButton.addEventListener("click", enableCam);
    enableWebcamButton.innerText = "Start de Game!"
    console.log("poselandmarker is ready!")
};


function fillList(){
    let select = document.getElementById("poseList");
    select.options.length = 0;
    for (let i = 0; i < poseArray.length; i++){
        let option = document.createElement("option");
        option.text = poseArray[i].label;
        option.value = i;
        select.add(option);
    }

}
document.getElementById("poseList").addEventListener("change", function (){
    console.log(typeof poseArray[this.value]);
    currentPose = poseArray[this.value];

    console.log(currentPose);
    console.log(`Current selected pose: ${currentPose.label}`);
})
function addNewPose(){
    const newPoseName = prompt("Please enter the name of the new pose");
    console.log(newPoseName);
    const newPose = {
        label: newPoseName,
        data:[]
    }
    poseArray.push(newPose);
    fillList();

    let select = document.getElementById("poseList");
    select.value = poseArray.indexOf(newPose);
    currentPose = newPose;
}

/********************************************************************
 // Continuously grab image from webcam stream and detect it.
 ********************************************************************/
function enableCam(event) {
    console.log("start the webcam")
    if (!poseLandmarker) {
        console.log("Wait! poseLandmaker not loaded yet.");
        return;
    }
    webcamRunning = true;
    enableWebcamButton.innerText = "Predicting";
    enableWebcamButton.disabled = true

    const constraints = {
        video: {
            width: { exact: 480 },
            height: { exact: 270 }
        }
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", async() => {
            canvasElement.style.height = videoHeight;
            canvasElement.style.width = videoWidth;
            video.style.height = videoHeight;
            video.style.width = videoWidth;
            predictWebcam();
        });
    });
}
// ********************************************************************
// detect poses!!
// ********************************************************************
async function predictWebcam() {
    let startTimeMs = performance.now();
    poseLandmarker.detectForVideo(video, performance.now(), (result) => drawPose(result));

    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}
// ********************************************************************
// draw the poses or log them in the console
// ********************************************************************
function drawPose(result) {
    // console.log(result.landmarks)
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    // // log de coordinaten
    // // console.log(result)
    // // teken de coordinaten in het canvas
    for (const landmark of result.landmarks) {
        drawingUtils.drawLandmarks(landmark, { radius: 3 });
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
    }
    let landmarks = result.landmarks[0];


    let cleandata = [];

    // landmarks naar een simpeler array zetten

    if (tickUpdate && landmarks){
        for (let i = 0; i < landmarks.length; i++)
        {
            cleandata.push(landmarks[i].x, landmarks[i].y, landmarks[i].z);
            tickUpdate = false
        }
        if(recording)
        recordPose(cleandata)
        console.log(cleandata)
        console.log(landmarks)

    }
}


startApp()
