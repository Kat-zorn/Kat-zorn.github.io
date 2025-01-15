function onBodyLoad() {
    writeBlogs("blog_index", "BlogList");
    writeBlogs("pinned_index", "PinnedList");
}

function instantiateBlogEntry(name) {
    return "<a href=\"md-render?filename=../blogs/" + name + "\">" + name + "</a><br>\n";
}

function loadBlogs(filename) {
    let xmlHttp = new XMLHttpRequest();
    xmlHttp.open('GET', filename, false); // false for synchronous request
    xmlHttp.send(null);
    const ret = xmlHttp.responseText;
    return ret.split('\n');
}

function writeBlogs(filename, listID) {
    const list = document.getElementById(listID);
    let blogs = loadBlogs(filename);
    blogs.forEach((name) => {
        const entry = instantiateBlogEntry(name);
        list.innerHTML += entry;
    });
}