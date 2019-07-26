let charRNN;

let temperature = .88;
let currentParagraph;
let maxParagraphCount = 1;
let countryname;
let addButton;

function setup() {

	noCanvas();


	charRNN = ml5.charRNN('models/combined', () => {
		console.log("model ready");
	});


	addButton = select("#add");
	addButton.mousePressed(()=>{

		maxParagraphCount++;
		loopRNN();
	})
}

function draw()
{
	if(!currentParagraph && countryname != null)
		addButton.show();
	else
		addButton.hide();
}


async function loopRNN() {

	await newParagraph();

	while (currentParagraph) {

		let text = currentParagraph.html();
		finished = text.length > 100 && text.charAt(text.length - 1) === '.';

		if (finished) {
			if (select('#result').elt.childElementCount < maxParagraphCount) {
				await newParagraph();
			} else {
				currentParagraph = null;
				break;
			}
		}

		await predict();
	}
}

async function newParagraph() {

	let seed = "The constitution of " + countryname + ", Art. 1\n" + countryname + " ";
	currentParagraph = createElement("li");
	currentParagraph.parent('#result');
	charRNN.reset();
	await charRNN.feed(seed);
	addText(countryname + " ")
}

async function predict() {
	let next = await charRNN.predict(temperature);
	//console.log(next)
	await charRNN.feed(next.sample);
	addText(next.sample);
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
let sketchClassProbabilities= [];
let sketchClassNames = [];
/*
prepare the drawing canvas
*/

$(document).ready(()=>{

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

async function load()
{

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
	}).done((data)=>{

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
		sketchClassProbabilities = findTopValues(pred, indices,5)
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

	let newCountryName = sketchClassNames[0];

	if(newCountryName != countryname)
	{
		countryname = "The Republic of the "+ newCountryName;
		select("#result").html("");

		let elements = document.querySelectorAll(".countryname-text");
		elements.forEach((el) => {
			el.innerHTML = "Constitution of " + countryname
		});
		loopRNN();
	}

}