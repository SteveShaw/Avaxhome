//viewportSize being the actual size of the headless browser

//the clipRect is the portion of the page you are taking a screenshot of
//page.clipRect = { top: 0, left: 0, width: 1024, height: 768 };
//the rest of the code is the same as the previous example
var wp = require('webpage')
var page = wp.create();
page.settings.resourceTimeout = 3000;
//page.settings.javascriptEnabled = false;
page.settings.loadImages = false;

var fs = require('fs')
var page_root = 'http://bt.aisex.com/bt/thread.php?fid=4&page='

var num_of_pages = 10;
var page_urls = []

var file_path = 'C:/StudyProj/mybase.html';
var html_path = 'file:///' + file_path;



function parse_page()
{
	console.log('Start Parsing');
	
	var mv_urls = page.evaluate(
		function()
		{
			return [].map.call(
				document.querySelectorAll('tr.tr3 h3 a'),
				function(a_tag)
				{
					return a_tag.getAttribute('href');
				}
			)
		}
	);
	
	console.log('End Parsing')
	
	console.log('found links='+mv_urls.length);
	// for(var i = 0; i< my_urls.length; ++i)
	// {
		// console.log(my_urls[i]);
	// }
	
	
	
	phantom.exit();
}

// // function OpenTopPage()
// // {
    // // var bookUrls = page.evaluate( 
        // // function()
        // // {
           // // return [].map.call(
            // // document.querySelectorAll('div.col-md-12.article a.title-link'), 
                    // // function(a) 
                   // // {
                      // // return a.getAttribute('href') 
                   // // }
            // // )
        // // }
    // // )
    
    // // var num_links = bookUrls.length 
      
    // // for(var i = 0;i < num_links;++i)
    // // {
       // // var bookUrl = bookUrls[i]
       // // if(bookUrl.length > 0 )
       // // {
           // // urls.push(url_root + bookUrls[i])   
       // // }
    // // }
    
    // // console.log( 'found books = ' + urls.length )
// // }

// function OpenNextTopPage()
// {
    // page.close()
    
    // ++topPageCount;
    
    // if(topPageCount >= num_of_pages)
    // {
        // console.log(' Final Top page is reached, will visit each child page to get book ')
        // count = 0;

        // fs.write("books.txt", urls.join('\n'), {mode: 'w', charset: 'UTF-8'})
        
        // phantom.exit()
    // }
    
    // console.log(' page count = ' + topPageCount)
    
    // page = wp.create()
    
    // page.open(topPageUrls[topPageCount], function( status )
        // {
           // console.log("visit "+ topPageUrls[topPageCount] + ' Status:' + status )
           
           // OpenTopPage()
           
           // window.setTimeout(OpenNextTopPage, 500)          
        // }
    // )
// }


// phantom.onError = function(msg, trace) {
  // var msgStack = ['PHANTOM ERROR: ' + msg];
  // if (trace && trace.length) {
    // msgStack.push('TRACE:');
    // trace.forEach(function(t) {
      // msgStack.push(' -> ' + (t.file || t.sourceURL) + ': ' + t.line + (t.function ? ' (in function ' + t.function +')' : ''));
    // });
  // }
  // console.error(msgStack.join('\n'));
  // phantom.exit(1);
// };

page.open( html_path, function(status) {
  
	console.log(page.title)
	//console.log(html_path)
    parse_page();
})




