
function onBodyLoad() {
    let blogEntryList = document.getElementById("BlogList");
    var directory = "blogs/";
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open('GET', directory, false); // false for synchronous request
    xmlHttp.send(null);
    let ret = xmlHttp.responseText;
    let blogFiles = ret.split('\n');
    blogFiles = blogFiles.slice(1, -1);
    console.log(blogFiles);
    let blogs = blogFiles.map(
        (entry) => entry.split(' ')[1]
    );
    blogs.forEach((name) => {
        const entry = instantiateBlogEntry(name);
        blogEntryList.innerHTML += entry;
    });
    return 0;
}

function instantiateBlogEntry(name) { 
    return "<a href=\"blogs/" + name + "\">" + name + "</a><br>\n";
}