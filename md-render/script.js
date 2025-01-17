function renderMarkdown() {
    let filename = getArgument('filename');
    let markdown = loadMarkdown(filename);
    let converter = new showdown.Converter();
    converter.setOption('strikethrough', true);
    converter.addExtension(showdownKatex(
        {
            output: 'html',
            throwOnError: false,
            displayMode: true
         }))
    let html = converter.makeHtml(markdown);
    let body = document.getElementById('MarkdownBody');
    body.innerHTML += html;
    document.getElementsByClassName("katex-html").innerHTML += "hidden=true";
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