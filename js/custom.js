function GenerateRandomString(length)
{
    var src = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890";
    var tgt = "";

    for(var i = 0; i < length; i++)
    {
        tgt += src.charAt(Math.floor(Math.random() * src.length));
    }

    return tgt;
}

const OKICON = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
OKICON.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
OKICON.setAttribute('role', 'img');
OKICON.setAttribute('width', '24');
OKICON.setAttribute('height', '24');
OKICON.setAttribute('viewBox', '0 0 24 24');
OKICON.classList.add('code-icon');

var OKTag1 = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
OKTag1.setAttribute('points', "4 13 9 18 20 7");

OKICON.appendChild(OKTag1);



const COPYICON = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
COPYICON.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
COPYICON.setAttribute('role', 'img');
COPYICON.setAttribute('width', '24');
COPYICON.setAttribute('height', '24');
COPYICON.setAttribute('viewBox', '0 0 24 24');
COPYICON.setAttribute('aria-labelledby', 'copy');
COPYICON.classList.add('code-icon');

var COPYtag1 = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
COPYtag1.setAttribute('width', '12');
COPYtag1.setAttribute('height', '14');
COPYtag1.setAttribute('x', '8');
COPYtag1.setAttribute('y', '7');

var COPYtag2 = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
COPYtag2.setAttribute('points', "16 3 4 3 4 17");

COPYICON.appendChild(COPYtag1);
COPYICON.appendChild(COPYtag2);


var THEMEDICT = {'light':'fa-sun','dark':'fa-moon'};




var NAVIGATION_LIST = null;
var OUTLINES = null;
var ACTIVE_NAV_ID = 0;

// proved useful for single case, could further expand.
function GenerateNavigation(){
    var sidelist = document.getElementById('nav-list')

    OUTLINES = document.querySelectorAll('h2,h3');

    var node = null;
    var anchor = null;

    for (var i = 0;i < OUTLINES.length; i++)
    {
        var outline = OUTLINES[i];
        if (outline.tagName == 'H2')
        {
            node = document.createElement('div');
            node.className = "nav-div nav-level2";
            anchor = document.createElement('a');
            if (outline.id == "")
            {
                outline.id = GenerateRandomString(10);
            }
            anchor.href = '#' + outline.id;
            node.appendChild(anchor);
            anchor.appendChild(document.createTextNode(outline.innerText));
        }
        else if (outline.tagName == 'H3')
        {
            node = document.createElement('div');
            node.className = "nav-div nav-level3";
            anchor = document.createElement('a');
            if (outline.id == "")
            {
                outline.id = GenerateRandomString(10);
            }
            anchor.href = '#' + outline.id;
            node.appendChild(anchor);
            anchor.appendChild(document.createTextNode(outline.innerText));
        }
        // console.log(node);
        sidelist.appendChild(node);
    }

    NAVIGATION_LIST = document.querySelectorAll('.nav-div');

    NAVIGATION_LIST[ACTIVE_NAV_ID].classList.add('nav-div-active');
}

function LocateNav(){
    OUTLINES = document.querySelectorAll('h2,h3');
    var cnt = 0;
    var atend = true;
    for (let i = 0;i < OUTLINES.length; i++)
    {
        var outline = OUTLINES[i];
        if (outline.getBoundingClientRect().top > document.documentElement.clientTop + 1)
        {
            cnt = i > 1 ? i - 1 : 0;
            atend = false;
            break;
        }
    }
    if (atend)
    {
        cnt = OUTLINES.length - 1;
    }
    if (cnt != ACTIVE_NAV_ID)
    {
        NAVIGATION_LIST[ACTIVE_NAV_ID].classList.remove('nav-div-active');
        ACTIVE_NAV_ID = cnt;
        NAVIGATION_LIST[ACTIVE_NAV_ID].classList.add('nav-div-active');
    }
}


function FormatCodes()
{
    var codes = document.querySelectorAll('code');

    for(let i = 0; i < codes.length; i++)
    {
        var code = codes[i];
        var txt = code.innerHTML;

        //console.log('>',txt);

        txt = txt.replace(/&amp;/g, "&");

        //console.log(txt);

        while(txt.startsWith('\n'))
        {
            txt = txt.substring(1);
        }

        code.innerHTML = txt;
    }
}

function CopyCode()
{
    //var codeboard = document.querySelectorAll('pre')[2];

    var codeboard = this.parentNode; //<pre>

    codeboard = codeboard.querySelector('code');

    var codelines = codeboard.getElementsByTagName('tr');

    //console.log(codelines);

    var code = "";

    for(let line of codelines)
    {
        var codeline = line.getElementsByClassName('hljs-ln-code');

        for(let c of codeline)
        {
            code += c.innerText;
        }
        code += '\n';
    }

    navigator.clipboard.writeText(code);

    var copydiv = codeboard.parentNode.firstChild;
    copydiv.removeChild(copydiv.firstChild);
    copydiv.appendChild(OKICON.cloneNode(true));
}

function AddCopyIcon()
{
    /* draw a div icon*/
    var copydiv = document.createElement('div');
    copydiv.className = 'clip';

    copydiv.appendChild(COPYICON);

    var pres = document.querySelectorAll('pre');
    
    for(let i = 0; i < pres.length; i++)
    {
        let fChild = pres[i].firstChild;
        pres[i].insertBefore(copydiv.cloneNode(true), fChild);
        pres[i].firstChild.addEventListener('click', CopyCode);
    }
}




// function TestFunc(){
//     var frames = document.querySelectorAll('.embed-frame');
//     console.log(frames);

//     var frame = frames[0];

//     frame.style.top = '120px';


// }

// if (window.self !== window.top)
// {
//     var frames = document.querySelectorAll('iframe');

//     console.log(frames);
//     for(let frame in frames)
//     {
//         frame.style.display = 'none';
//         frame.style.setProperty('display', 'none', 'important');
//     }
// }

function SetTheme(theme)
{
    var html = document.querySelector('html');
    html.setAttribute('data-bs-theme', theme);
    localStorage.setItem('theme', theme);
    var icon = document.getElementById('theme-control');
    icon.className = '';
    icon.classList.add('fa-regular', THEMEDICT[theme]);
}

function InitTheme(){
    var theme = localStorage.getItem('theme');
    if (theme == null)
    {
        var hour = new Date().getHours();
        if (hour >= 6 && hour < 18)
        {
            SetTheme('light');
        }
        else
        {
            SetTheme('dark');
        }
    }
    else
    {
        SetTheme(theme);
    }
}

function SwitchTheme()
{
    var theme = localStorage.getItem('theme');
    if (theme == 'light')
    {
        SetTheme('dark');
    }
    else
    {
        SetTheme('light');
    }
}

function LoadDocumentContent(href,callback)
{
    // console.log(href);

    if (href == null)
    {
        callback();
        return;
    }

    let xhr = new XMLHttpRequest();

    xhr.open('GET', href, true);

    content = document.querySelector('.content-body')

    xhr.onreadystatechange = function(){
        if (xhr.readyState == 4 && xhr.status == 200)
        {
            // console.log(xhr.responseText)

            var text = xhr.responseText;

            //text = text.replace('<', '&lt;').replace('>', '&gt;');

            content.innerHTML = text;
            //console.log(content.innerHTML)
            callback();
            FormatCodes();
            AddCopyIcon();
            GenerateNavigation();
        }
    }
    xhr.send();
}


// document.addEventListener('DOMContentLoaded', FormatCodes);
// document.addEventListener('DOMContentLoaded', GenerateNavigation);
// document.addEventListener('DOMContentLoaded', AddCopyIcon);
document.addEventListener('DOMContentLoaded', InitTheme);
// document.addEventListener('DOMContentLoaded', TestFunc)
document.addEventListener('scroll', LocateNav);

// CopyCode();

