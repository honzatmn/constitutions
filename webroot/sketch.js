
let currentParagraph;
let maxParagraphCount = 3;
let maxPreambleCount = 3;
let countryname;
let countryTag;
let addButton;
let isRunning;
let canceled;

let charRNNPreambles;
let charRNNArticles;

function setup() {

	noCanvas();


	charRNNPreambles = ml5.charRNN('models/preambles', () => {
		console.log("model ready");
	});

	charRNNArticles = ml5.charRNN('models/combined', () => {
		console.log("model ready");
	});


	addButton = select("#add");
	addButton.mousePressed(() => {

		maxParagraphCount++;
	})
}

function draw() {

	if (!isRunning && countryname )
		addButton.show();
	else
		addButton.hide();

	let preambleCount = select('#preambles').elt.childElementCount;
	let articleCount = select('#articles').elt.childElementCount;
	let doPreambles = preambleCount < maxPreambleCount;
	let target = doPreambles? "#preambles" : "#articles";
	let maxCount =  doPreambles ? maxPreambleCount : maxParagraphCount;
	let currCount =  doPreambles ? preambleCount : articleCount;
	if(!isRunning && countryname && currCount < maxCount){

		let rnn = doPreambles ? charRNNPreambles : charRNNArticles;
		select('#preambles-title').html("Preambles");

		let temperature = doPreambles ? 0.9 : .88;
		startParagraphLoop(target,rnn,temperature);
	}

}


async function startParagraphLoop(target,charRNN,temperature) {

	console.log("prediction loop")
	if(isRunning){

		return;
	}

	isRunning = true;
	canceled = false;

	currentParagraph = createElement("li");
	currentParagraph.html(name)
	currentParagraph.parent(target);

	let ucCountryName = jsUcfirst(countryname);
	let seed = "The country";

	let lastChar = null;
	await charRNN.reset();
	await charRNN.feed(seed);
	let next = await charRNN.predict(temperature);
	lastChar = next.sample;
	await charRNN.feed(lastChar);

	addText(ucCountryName);

	while (currentParagraph && !(currentParagraph.html().length > 100 && lastChar === '.')  && !canceled) {

		let next = await charRNN.predict(temperature);

		next = await charRNN.predict(temperature);

		next = await charRNN.predict(temperature);
		lastChar = next.sample;
		await charRNN.feed(lastChar);
		addText(lastChar);
	}
	isRunning = false;
}
function delay(ms)
{
	return new Promise(resolve => setTimeout(resolve, ms));
}

function jsUcfirst(string)
{

	return string.charAt(0).toUpperCase() + string.slice(1);

}
function addText(text) {
	currentParagraph.html(currentParagraph.html() + text);
}


/*
variables
*/
var model;
var canv;
var classNames = [];
var coords = [];
var mousePressed = false;
let sketchClassProbabilities = [];
let sketchClassNames = [];
/*
prepare the drawing canvas
*/

$(document).ready(() => {

	canv = window._canvas = new fabric.Canvas('canvas');
	canv.backgroundColor = '#ffffff';
	canv.isDrawingMode = 0;
	canv.freeDrawingBrush.color = "black";
	canv.freeDrawingBrush.width = 10;
	canv.renderAll();
	//setup listeners
	canv.on('mouse:up', function (e) {
		getFrame();
		mousePressed = false
	});
	canv.on('mouse:down', function (e) {
		mousePressed = true
	});
	canv.on('mouse:move', function (e) {
		recordCoor(e)
	});

	load();
})

async function load() {

	//load the model
	model = await tf.loadLayersModel('models/sketch-detection2/model.json')

	//warm up
	model.predict(tf.zeros([1, 28, 28, 1]))

	//allow drawing on the canvas
	canv.isDrawingMode = 1;
	$('button').prop('disabled', false);

	//load the class names

	await $.ajax({
		url: 'models/sketch-detection2/class_names.txt',
		dataType: 'text',
	}).done((data) => {

		const lst = data.split(/\n/)
		for (var i = 0; i < lst.length - 1; i++) {
			let symbol = lst[i]
			classNames[i] = symbol
		}
	});
}

/*
record the current drawing coordinates
*/
function recordCoor(event) {
	var pointer = canv.getPointer(event.e);
	var posX = pointer.x;
	var posY = pointer.y;

	if (posX >= 0 && posY >= 0 && mousePressed) {
		coords.push(pointer)
	}
}

/*
get the best bounding box by trimming around the drawing
*/
function getMinBox() {
	//get coordinates
	var coorX = coords.map(function (p) {
		return p.x
	});
	var coorY = coords.map(function (p) {
		return p.y
	});

	//find top left and bottom right corners
	var min_coords = {
		x: Math.min.apply(null, coorX),
		y: Math.min.apply(null, coorY)
	}
	var max_coords = {
		x: Math.max.apply(null, coorX),
		y: Math.max.apply(null, coorY)
	}

	//return as strucut
	return {
		min: min_coords,
		max: max_coords
	}
}

/*
get the prediction
*/
function getFrame() {
	//make sure we have at least two recorded coordinates
	if (coords.length >= 2) {

		//get the image data from the canvas
		const mbb = getMinBox()

		//get image data according to dpi
		const dpi = window.devicePixelRatio
		const imgData = canv.contextContainer.getImageData(mbb.min.x * dpi, mbb.min.y * dpi,
			(mbb.max.x - mbb.min.x) * dpi, (mbb.max.y - mbb.min.y) * dpi);

		const preprocessedData = tf.tidy(() => {
			//convert to a tensor
			let tensor = tf.browser.fromPixels(imgData, numChannels = 1)

			//resize
			const resized = tf.image.resizeBilinear(tensor, [28, 28]).toFloat()

			//normalize
			const offset = tf.scalar(255.0);
			const normalized = tf.scalar(1.0).sub(resized.div(offset));

			//We add a dimension to get a batch shape
			const batched = normalized.expandDims(0)
			return batched
		});

		//get the prediction
		const pred = model.predict(preprocessedData).dataSync()

		//find the top 5 predictions
		let indices = findIndicesOfMax(pred, 5)
		sketchClassProbabilities = findTopValues(pred, indices, 5)
		sketchClassNames = getClassNames(indices)
	}
}


/*
get the the class names
*/
function getClassNames(indices) {
	var outp = []
	for (var i = 0; i < indices.length; i++)
		outp[i] = classNames[indices[i]]
	return outp
}

/*
get indices of the top probs
*/
function findIndicesOfMax(inp, count) {
	var outp = [];
	for (var i = 0; i < inp.length; i++) {
		outp.push(i); // add index to output array
		if (outp.length > count) {
			outp.sort(function (a, b) {
				return inp[b] - inp[a];
			}); // descending sort the output array
			outp.pop(); // remove the last index (index of smallest element in output array)
		}
	}
	return outp;
}

/*
find the top 5 predictions
*/
function findTopValues(inp, indices, count) {
	var outp = [];
	// show 5 greatest scores
	for (var i = 0; i < indices.length; i++)
		outp[i] = inp[indices[i]];
	return outp

}

/*
clear the canvs
*/
function erase() {
	canv.clear();
	canv.backgroundColor = '#ffffff';
	coords = [];
}

function submit() {

	let newCountryTag = sketchClassNames[0];

	if (newCountryTag != countryTag) {

		countryTag = newCountryTag;
		countryname = getNiceCountryName(countryTag);
		select("#preambles").html("");
		select("#articles").html("");

		canceled = true;

		let elements = document.querySelectorAll(".countryname-text");
		elements.forEach((el) => {
			el.innerHTML = "Constitution of " + countryname
		});
	}

}

function getNiceCountryName(tag) {

	return prefixes[floor(random(0,prefixes.length))] + niceNames[tag];
}

let prefixes = [
	"the ",
	"the Republic of the ",
	"the United States of the ",
	"the Kingdom of the ",
]

let niceNames = {"screwdriver":"Screwdriver",
	"wristwatch":"Wristwatch",
	"butterfly":"Butterfly",
	"sword":"Sword",
	"cat":"Cat",
	"shorts":"Shorts",
	"eyeglasses":"Eyeglasses",
	"lollipop":"Lollipop",
	"baseball":"Baseball",
	"traffic_light":"Traffic Light",
	"sun":"Sun",
	"helmet":"Helmet",
	"bridge":"Bridge",
	"alarm_clock":"Alarm Clock",
	"drums":"Drums",
	"book":"Book",
	"broom":"Broom",
	"fan":"Fan",
	"scissors":"Scissors",
	"cloud":"Cloud",
	"tent":"Tent",
	"clock":"Clock",
	"headphones":"Headphones",
	"bicycle":"Bicycle",
	"stop_sign":"Stop Sign",
	"table":"Table",
	"donut":"Donut",
	"umbrella":"Umbrella",
	"smiley_face":"Smiley Face",
	"pillow":"Pillow",
	"bed":"Bed",
	"saw":"Saw",
	"light_bulb":"Light Bulb",
	"shovel":"Shovel",
	"bird":"Bird",
	"syringe":"Syringe",
	"coffee_cup":"Coffee Cup",
	"moon":"Moon",
	"ice_cream":"Ice Cream",
	"moustache":"Moustache",
	"cell_phone":"Cell Phone",
	"pants":"Pants",
	"anvil":"Anvil",
	"radio":"Radio",
	"chair":"Chair",
	"star":"Star",
	"door":"Door",
	"face":"Face",
	"mushroom":"Mushroom",
	"tree":"Tree",
	"rifle":"Rifle",
	"camera":"Camera",
	"lightning":"Lightning",
	"flower":"Flower",
	"basketball":"Basketball",
	"wheel":"Wheel",
	"hammer":"Hammer",
	"hat":"Hat",
	"knife":"Knife",
	"diving_board":"Diving Board",
	"square":"Square",
	"cup":"Cup",
	"mountain":"Mountain",
	"apple":"Apple",
	"spoon":"Spoon",
	"key":"Key",
	"pencil":"Pencil",
	"line":"Line",
	"ladder":"Ladder",
	"triangle":"Triangle",
	"t-shirt":"T-Shirt",
	"dumbbell":"Dumbbell",
	"microphone":"Microphone",
	"snake":"Snake",
	"sock":"Sock",
	"suitcase":"Suitcase",
	"laptop":"Laptop",
	"paper_clip":"Paper Clip",
	"rainbow":"Rainbow",
	"candle":"Candle",
	"bread":"Bread",
	"spider":"Spider",
	"envelope":"Envelope",
	"circle":"Circle",
	"power_outlet":"Power Outlet",
	"tooth":"Tooth",
	"hot_dog":"Hot Dog",
	"frying_pan":"Frying Pan",
	"bench":"Bench",
	"ceiling_fan":"Ceiling Fan",
	"tennis_racquet":"Tennis Racquet",
	"car":"Car",
	"beard":"Beard",
	"axe":"Axe",
	"baseball_bat":"Baseball Bat",
	"pizza":"Pizza",
	"grapes":"Grapes",
	"eye":"Eye",
	"cookie":"Cookie",
	"airplane":"Airplane"
}