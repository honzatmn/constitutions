let countryname;
let countryTag;
let addButton;

let rnns = [];
let preamblesGenerator;
let articlesGenerator;
let anthemRNN;
let anthem;

function setup() {

	noCanvas();

	anthemRNN = ml5.charRNN("models/anthems", () => {
		console.log("anthems ready");
	});
	mottoGenerator = new RNNGenerator('models/latin-phrases', "Non", " ", 30, "#motto", "#motto-wrapper", "span", 1, 0.2);
	preamblesGenerator = new RNNGenerator('models/preambles', null, ".", 100, "#preambles", "#preambles-wrapper", "li", 2, 0.95);
	articlesGenerator = new RNNGenerator('models/articles-combined-30', null, ".", 100, "#articles", "#articles-wrapper", "li", 3, 0.88);

	rnns.push(mottoGenerator);
	rnns.push(preamblesGenerator);
	rnns.push(articlesGenerator);

	addButton = select("#add");
	addButton.mousePressed(() => {

		articlesGenerator.maxCount++;
	})

	setCountryTag(null);
}

function draw() {

	if (!isAnyRunning() && countryname)
		addButton.show();
	else
		addButton.hide();

	if (!isAnyRunning() && countryname) {

		let currGenerator = null;
		for (let i = 0; i < rnns.length; i++) {

			let rnn = rnns[i];
			if (!rnn.isFinished()) {
				currGenerator = rnn;
				break;
			}
		}

		if (currGenerator) {

			select('#constitution').show();
			select('.info-wrapper').show();
			

			currGenerator.startParagraphLoop();
		}

	}

}

function isAnyRunning() {

	let isRunningAny = false;
	rnns.forEach(r => isRunningAny |= r.isRunning);

	return isRunningAny;
}


/*
variables
*/
var sketchModel;
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

	load();
})

async function load() {

	//load the model
	sketchModel = await tf.loadLayersModel('models/sketch-detection/model.json')

	console.log("sketch model loaded");
	//warm up
	sketchModel.predict(tf.zeros([1, 28, 28, 1]))

	//allow drawing on the canvas
	canv.isDrawingMode = 1;
	$('button').prop('disabled', false);

	//load the class names

	await $.ajax({
		url: 'models/sketch-detection/class_names.txt',
		dataType: 'text',
	}).done((data) => {

		const lst = data.split(/\n/)
		for (var i = 0; i < lst.length - 1; i++) {
			let symbol = lst[i]
			classNames[i] = symbol
		}
	});

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
		const pred = sketchModel.predict(preprocessedData).dataSync()

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

	sketchClassNames = [];
	setCountryTag(null);
}

function submit() {

	//canv.backgroundColor = '#' + Math.floor(Math.random() * 16777215).toString(16);
	//canv.renderAll();
	if (sketchClassNames.length < 1)
		return;

	let newCountryTag = sketchClassNames[0];
	setCountryTag(newCountryTag);

	var c = document.getElementById("canvas");
	var ctx = c.getContext("2d");
	var my_gradient = ctx.createLinearGradient(0, 0, 180, Math.floor(Math.random() * 360));
	ctx.globalCompositeOperation = 'multiply';
	my_gradient.addColorStop(0, random_rgba());
	my_gradient.addColorStop(0.5, random_rgba());
	my_gradient.addColorStop(1, random_rgba());
	ctx.fillStyle = my_gradient;
	ctx.fillRect(0, 0, 300, 300);

}

function setCountryTag(newCountryTag) {

	if (newCountryTag === countryTag) {
		return;
	}
	countryTag = newCountryTag;
	countryname = countryTag ? getNiceCountryNameWithPrefix(countryTag) : null;

	updateNeighbours();
	select("#countryname-text").html(countryTag ? "Constitution of " + countryname : null);


	if (countryTag)
		select("#constitution").show();
	else
		select("#constitution").hide();

	if (countryTag)
		select("#info").hide();
	else
		select("#info").show();


	rnns.forEach(r => r.reset());


	generateAnthem(countryname);
}

function random_rgba() {
    var o = Math.round, r = Math.random, s = 255;
    return 'rgba(' + o(r()*s) + ',' + o(r()*s) + ',' + o(r()*s) + ',' + r().toFixed(1) + ')';
}

async function generateAnthem(seed) {
	anthem = "";

	if (!seed)
		return;

	let temperature = 0.5;
	let anthemlength = 40;
	await anthemRNN.reset();
	await anthemRNN.feed(seed);
	for (let i = 0; i < anthemlength; i++) {

		let next = await anthemRNN.predict(temperature);
		let lastChar = next.sample;
		anthem += lastChar;
		await anthemRNN.feed(lastChar);
	}

	playAnthem();
}

function playAnthem()
{
	playSequenceStr(anthem);
}

function updateNeighbours() {

	let neighbours = select("#neighbours");
	neighbours.html("");

	let neighboursTitle = select("#neighbours-title");
	let neighboursTitleContent = countryTag ? getNiceGroupName(getNiceCountryName(countryTag)) + ":" : "";
	neighboursTitle.html(neighboursTitleContent)

	for (let i = 1; i < sketchClassNames.length; i++) {

		let e = sketchClassNames[i];
		let niceName = getNiceCountryNameWithPrefix(e);
		let el = createElement("li", niceName);
		el.parent(neighbours);
	}
	let underTitle = "Based on the top guesses of what you drew";
}

function getNiceGroupName(name) {

	return groupPrefixes[floor(random(0, groupPrefixes.length))] + name;
}

function getNiceCountryNameWithPrefix(tag) {

	return prefixes[floor(random(0, prefixes.length))] + getNiceCountryName(tag);
}

function getNiceCountryName(tag) {

	return "the " + niceNames[tag];
}

let groupPrefixes = [
	"Union of ",
	"Commonwealth of ",
	"United States of ",
	"Federal Union of "
]

let prefixes = [
	"",
	"",
	"the Republic of ",
	"the Kingdom of ",
	"the Sultanate of ",
	"the Empire of ",
	"the State of ",
	"the Dynasty of ",
	"the Nation of ",
	"the Country of "
]

let niceNames = {
	"screwdriver": "Screwdriver",
	"wristwatch": "Wristwatch",
	"butterfly": "Butterfly",
	"sword": "Sword",
	"cat": "Cat",
	"shorts": "Shorts",
	"eyeglasses": "Eyeglasses",
	"lollipop": "Lollipop",
	"baseball": "Baseball",
	"traffic_light": "Traffic Light",
	"sun": "Sun",
	"helmet": "Helmet",
	"bridge": "Bridge",
	"alarm_clock": "Alarm Clock",
	"drums": "Drums",
	"book": "Book",
	"broom": "Broom",
	"fan": "Fan",
	"scissors": "Scissors",
	"cloud": "Cloud",
	"tent": "Tent",
	"clock": "Clock",
	"headphones": "Headphones",
	"bicycle": "Bicycle",
	"stop_sign": "Stop Sign",
	"table": "Table",
	"donut": "Donut",
	"umbrella": "Umbrella",
	"smiley_face": "Smiley Face",
	"pillow": "Pillow",
	"bed": "Bed",
	"saw": "Saw",
	"light_bulb": "Light Bulb",
	"shovel": "Shovel",
	"bird": "Bird",
	"syringe": "Syringe",
	"coffee_cup": "Coffee Cup",
	"moon": "Moon",
	"ice_cream": "Ice Cream",
	"moustache": "Moustache",
	"cell_phone": "Cell Phone",
	"pants": "Pants",
	"anvil": "Anvil",
	"radio": "Radio",
	"chair": "Chair",
	"star": "Star",
	"door": "Door",
	"face": "Face",
	"mushroom": "Mushroom",
	"tree": "Tree",
	"rifle": "Rifle",
	"camera": "Camera",
	"lightning": "Lightning",
	"flower": "Flower",
	"basketball": "Basketball",
	"wheel": "Wheel",
	"hammer": "Hammer",
	"hat": "Hat",
	"knife": "Knife",
	"diving_board": "Diving Board",
	"square": "Square",
	"cup": "Cup",
	"mountain": "Mountain",
	"apple": "Apple",
	"spoon": "Spoon",
	"key": "Key",
	"pencil": "Pencil",
	"line": "Line",
	"ladder": "Ladder",
	"triangle": "Triangle",
	"t-shirt": "T-Shirt",
	"dumbbell": "Dumbbell",
	"microphone": "Microphone",
	"snake": "Snake",
	"sock": "Sock",
	"suitcase": "Suitcase",
	"laptop": "Laptop",
	"paper_clip": "Paper Clip",
	"rainbow": "Rainbow",
	"candle": "Candle",
	"bread": "Bread",
	"spider": "Spider",
	"envelope": "Envelope",
	"circle": "Circle",
	"power_outlet": "Power Outlet",
	"tooth": "Tooth",
	"hot_dog": "Hot Dog",
	"frying_pan": "Frying Pan",
	"bench": "Bench",
	"ceiling_fan": "Ceiling Fan",
	"tennis_racquet": "Tennis Racquet",
	"car": "Car",
	"beard": "Beard",
	"axe": "Axe",
	"baseball_bat": "Baseball Bat",
	"pizza": "Pizza",
	"grapes": "Grapes",
	"eye": "Eye",
	"cookie": "Cookie",
	"airplane": "Airplane"
}

function jsUcfirst(string) {

	return string.charAt(0).toUpperCase() + string.slice(1);

}

class RNNGenerator {
	constructor(model, seed, endChar, minLength, target, targetContainer, elementType, maxcount, temperature) {

		this.rnn = ml5.charRNN(model, () => {
			console.log("model " + model + " ready");
		});

		this.elementType = elementType;
		this.target = target;
		this.targetContainer = targetContainer,
			this.temperature = temperature;
		this.currentParagraph = null;
		this.isRunning = false;
		this.canceled = false;
		this.maxCount = maxcount;
		this.endChar = endChar;
		this.seed = seed;
		this.minLength = minLength;

		this.reset();
	}

	isFinished() {
		return select(this.target).elt.childElementCount >= this.maxCount;
	}

	reset() {

		select(this.target).html("");
		select(this.targetContainer).hide();
		this.canceled = true
		select('.info-wrapper').hide();
	}

	async startParagraphLoop() {

		if (this.isRunning) {

			return;
		}

		this.isRunning = true;
		this.canceled = false;

		this.currentParagraph = createElement(this.elementType);
		this.currentParagraph.html(name)
		this.currentParagraph.parent(this.target);

		select(this.targetContainer).show();

		let seed = this.seed != null ? this.seed : "The country";

		await this.rnn.reset();
		await this.rnn.feed(seed);
		let next = await this.rnn.predict(this.temperature);
		let lastChar = next.sample;
		await this.rnn.feed(lastChar);

		if (this.seed == null) {
			let ucCountryName = jsUcfirst(countryname);
			this.addText(ucCountryName);
		}

		while (this.currentParagraph && !(this.currentParagraph.html().length > this.minLength && lastChar === this.endChar) && !this.canceled) {

			let next = await this.rnn.predict(this.temperature);
			lastChar = next.sample;
			await this.rnn.feed(lastChar);
			this.addText(lastChar);
		}

		if (lastChar === " ") {
			let str = this.currentParagraph.html();
			str = str.substring(0, str.length - 1);
			this.currentParagraph.html(str);
		}
		this.isRunning = false;
	}

	addText(text) {
		this.currentParagraph.html(this.currentParagraph.html() + text);
	}

}