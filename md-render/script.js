function renderMarkdown() {
    let filename = getArgument('filename');
    let markdown = loadMarkdown(filename);
    let converter = new Markdown.Converter();
    let html = converter.makeHtml(markdown);
    let body = document.getElementById('MarkdownBody');
    body.innerHTML += html;
}

function getArgument(name) {
    let url = window.location.search;
    let params = url.split('&');
    params[0] = params[0].slice(1); // Remove the initial '?'
    let param = params.find((str, idx, obj) => str.startsWith(name + '='));
    let value = param.slice(name.length + 1);
    return value;
}

function loadMarkdown(filename) {
    let xmlHttp = new XMLHttpRequest();
    xmlHttp.open('GET', filename, false); // false for synchronous request
    xmlHttp.send(null);
    return xmlHttp.responseText;
}