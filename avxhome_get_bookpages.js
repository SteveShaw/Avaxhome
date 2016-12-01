//viewportSize being the actual size of the headless browser

//the clipRect is the portion of the page you are taking a screenshot of
//page.clipRect = { top: 0, left: 0, width: 1024, height: 768 };
//the rest of the code is the same as the previous example
var wp = require('webpage')
var page = wp.create();
var fs = require('fs')
// var page_root = 'http://avxhome.in/ebooks/programming_development/pages/'
// var page_root = 'http://avxhome.in/ebooks/economics_finances/pages/'
// var page_root = 'http://avxhome.in/ebooks/science_books/math/pages/'
var page_root = 'http://avxhome.in/ebooks/eLearning/pages/';
// var url = page_root + 1
var url_root = 'http://avxhome.in'
var urls = []
var count = 0
var topPageCount = 0;
var topPageUrls = []
var page_contents = []
var isTopPage = false

var num_of_pages = 50;
var start_page = 1;

// var saveDirPath = 'page_1/'
// fs.makeDirectory( saveDirPath )

for(var i = 0;i < num_of_pages; ++i)
{
    var idx = i + start_page;
    console.log( page_root+idx )
    topPageUrls.push(page_root + idx)
}

function OpenTopPage()
{
    var bookUrls = page.evaluate( 
        function()
        {
           return [].map.call(
            document.querySelectorAll('div.col-md-12.article a.title-link'), 
                    function(a) 
                   {
                      return a.getAttribute('href') 
                   }
            )
        }
    )
    
    var num_links = bookUrls.length 
      
    for(var i = 0;i < num_links;++i)
    {
       var bookUrl = bookUrls[i]
       if(bookUrl.length > 0 )
       {
           urls.push(url_root + bookUrls[i])   
       }
    }
    
    console.log( 'found books = ' + urls.length )
}

function OpenNextTopPage()
{
    page.close()
    
    ++topPageCount;
    
    if(topPageCount >= num_of_pages)
    {
        console.log(' Final Top page is reached, will visit each child page to get book ')
        count = 0;

        fs.write("books.txt", urls.join('\n'), {mode: 'w', charset: 'UTF-8'})
        
        phantom.exit()
    }
    
    console.log(' page count = ' + topPageCount)
    
    page = wp.create()
    
    page.open(topPageUrls[topPageCount], function( status )
        {
           console.log("visit "+ topPageUrls[topPageCount] + ' Status:' + status )
           
           OpenTopPage()
           
           window.setTimeout(OpenNextTopPage, 500)          
        }
    )
}


phantom.onError = function(msg, trace) {
  var msgStack = ['PHANTOM ERROR: ' + msg];
  if (trace && trace.length) {
    msgStack.push('TRACE:');
    trace.forEach(function(t) {
      msgStack.push(' -> ' + (t.file || t.sourceURL) + ': ' + t.line + (t.function ? ' (in function ' + t.function +')' : ''));
    });
  }
  console.error(msgStack.join('\n'));
  phantom.exit(1);
};

page.open( topPageUrls[topPageCount], function(status){
  
    console.log("visit "+ topPageUrls[topPageCount] + ' Status:' + status )
    
    OpenTopPage()
 
    window.setTimeout(OpenNextTopPage, 500)
})




