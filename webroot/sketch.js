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

	let generateButton = select("#generate");
	generateButton.mousePressed(
		() => {
			select("#result").html("");

			countryname = select("#countryname").value();
			let elements = document.querySelectorAll(".countryname-text");
			elements.forEach((el) => {
				el.innerHTML = "Constitution of " + countryname
			});
			loopRNN();
		});


	addButton = select("#add");
	addButton.mousePressed(()=>{

		maxParagraphCount++;
		loopRNN();
	})
}

function draw()
{
	let generateButton = select("#generate");
	generateButton.style("display", !currentParagraph && charRNN.ready ? "inline": "none");
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
