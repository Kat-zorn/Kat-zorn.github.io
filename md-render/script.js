function renderMarkdown() {
    let filename = getArgument('filename');
    let html = marked.render(filename);
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