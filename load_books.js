var fs = require('fs');
var wp = require('webpage')
var page = wp.create();

var content = fs.read('books.txt');

var title_filename = 'titles.txt'

var page_contents = []
var book_titles = []
var limits = 45
var save_count = 1

var links = content.split('\n')
console.log('total number of books = ' + links.length)
var count = 0;
var retry = false;

function PadZeros(num, size) {
    var s = num+"";
    while (s.length < size) s = "0" + s;
    return s;
}

function AddToPageContents( title, inner )
{
    page_contents.push('<div>')
    page_contents.push('<h2>'+ title + '</h2>')
    page_contents.push( inner )
    page_contents.push('</div>')
    page_contents.push('<hr>')
}

function SavePageContents()
{
    var outPath = PadZeros( save_count++, 3 ) + '.html'
    console.log('Save to "' + outPath + '"')
    fs.write(outPath, page_contents.join('\n'), {mode: 'w', charset: 'UTF-8'})
    page_contents = []
}

function RetrieveBookContent( )
{
    var titles = page.evaluate( 
        function()
        {
           return [].map.call(
            document.querySelectorAll('div.col-md-12.article h1.title-link'), 
                    function(obj) 
                   {
                      return obj.innerText
                   }
            )
        }
    )
    
    var a_links = page.evaluate(
        function() {
            return [].map.call(
            document.querySelectorAll('div.col-md-12.article a[target=_blank]'),
                function( obj )
                {
                    return obj.href;
                }
          )
        }
    );
    
    var descriptions = page.evaluate(
        function() {
            return [].map.call(
            document.querySelectorAll('div.col-md-12.article div.text'),
                function( obj )
                {
                    return obj.innerText;
                }
          )
        }
    );
    
    var inners = page.evaluate( 
        function()
        {
           return [].map.call(
            document.querySelectorAll('div.col-md-12.article'), 
                    function(obj) 
                   {
                      return obj.innerHTML
                   }
            )
        }
    )

    console.log( 'found number of titles = ' + titles.length )
    console.log( 'found links: ' + a_links.length)
    
    if( titles.length > 0 && inners.length > 0 )
    {
        console.log(titles[0])
        console.log('found inners length = ' + inners.length)
        AddToPageContents( titles[0], inners[0] )
        book_titles.push( [titles[0], a_links.join('\n'), descriptions[0]] )
    }
    else
    {
        console.log('Error: Missing div.col-md-12.article')
        // AddToPageContents( 'Untitled', '<div><h2>NONE</h2>'+'<a href="'+ url 
                            // +'" target="_blank"><span style="color: Red;font-size: 300\%">'
                            // +url+'</span></a>'+'</div>' );
		// book_titles.push( 'Error_Title', links[count], ' Error Access ' );
    }
}

function LoadPage()
{
    page.close()
    
    if( !retry )
    {
        ++count;
        
        if(count >= links.length)
        {
            SavePageContents()
            console.log('Save titles')
            var numBooks = book_titles.length
            var writeContents = []
            for(var i = 0;i<numBooks;++i)
            {
                writeContents.push('******')
                writeContents.push( i + ' <--> ' + book_titles[i][0] )
                writeContents.push( book_titles[i][1] )
                writeContents.push( book_titles[i][2] )
                writeContents.push('******')
            }
            fs.write(title_filename, writeContents.join('\n'), {mode: 'w', charset: 'UTF-8'})
            phantom.exit()
        }
        
        if( count % limits == 0 )
        {
            SavePageContents()
        }
    }
    else
    {
        console.log('Retry...')
    }
    
    page = wp.create()
    
    page.open(links[count], function(status) {
        
            console.log('current count = ' + count)
            console.log( 'visit: '+ links[count] + ' Status: ' + status )
            
            if( status != 'success' )
            {
                retry  = true;
            }
            else
            {
                retry = false;
            }
            
            RetrieveBookContent()
            window.setTimeout(LoadPage, 200);
        }
    )
}

page.open(links[count], function(status){
            
            console.log('current count = ' + count)
            console.log( 'visit: '+ links[count] + ' Status: ' + status )
            
            if( status != 'success' )
            {
                retry  = true;
            }
            else
            {
                retry = false;
            }
            
            RetrieveBookContent()
            window.setTimeout(LoadPage, 200);
        }
    )

