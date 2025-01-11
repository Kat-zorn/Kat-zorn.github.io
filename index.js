function onBodyLoad() {
    let blogEntryList = document.getElementById("BlogList");
    var blogIndexFile = "blog_index";
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open('GET', blogIndexFile, false); // false for synchronous request
    xmlHttp.send(null);
    let ret = xmlHttp.responseText;
    let blogs = ret.split('\n');
    blogs.forEach((name) => {
        const entry = instantiateBlogEntry(name);
        blogEntryList.innerHTML += entry;
    });
    return 0;
}

function instantiateBlogEntry(name) {
    return "<a href=\"blogs/" + name + "\">" + name + "</a><br>\n";
}