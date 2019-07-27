const request = require('request')
const writeFile = require('write')
const jsdom = require("jsdom");

const letters = ["A","B","C","D","E","F","G","H","I","L","M","N","O","P","Q","R","S","T","U","V"]
let pages = []

for (const letter of letters) {
    if (letter !== null) {
        pages.push("https://en.wikipedia.org/wiki/List_of_Latin_phrases_(" + letter + ")")
    }
}

let allLatinPhrases = ""

function getLinkContent (linkUrl) {

    return new Promise(function (resolve, reject) {
        const options = {
            url: linkUrl,
            timeout: 5000,
            gzip: true
        }

        request(options, function (error, response, body) {
            if (!error && response.statusCode === 200) {
                resolve({'link': linkUrl, 'content': body})
            } else {
                reject(error)
            }
        })
    }).then(function (data) {
        return data
    })
}


async function getAllPages() {
    let currentPage = 1
    let numberofPages = pages.length

    for (const page of pages) {


        if (currentPage > 3) {
            continue
        }

        let url = page

        await getLinkContent(url).then(function (file) {
            const content = file.content;

            console.log("Downloaded page", page, " – ", currentPage, "from", numberofPages);

            const dom = new jsdom.JSDOM(content);
            let phrases = dom.window.document.querySelectorAll(".wikitable td b")

            phrases.forEach(phrase => {
                allLatinPhrases += phrase.textContent + "\n"
            })

            currentPage += 1

        }).catch(function (e) {
            console.log(e)
        })
    }
}

getAllPages().then(function(){

    const path = 'data/latin-phrases.txt';

    writeFile(path, allLatinPhrases).catch(function (err) {
        console.log(err)
    })

    console.log("done")
})

