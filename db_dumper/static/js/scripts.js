/*!
    * Start Bootstrap - Grayscale v6.0.2 (https://startbootstrap.com/themes/grayscale)
    * Copyright 2013-2020 Start Bootstrap
    * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-grayscale/blob/master/LICENSE)
    */
    (function ($) {
    "use strict"; // Start of use strict

    // Smooth scrolling using jQuery easing
    $('a.js-scroll-trigger[href*="#"]:not([href="#"])').click(function () {
        if (
            location.pathname.replace(/^\//, "") ==
                this.pathname.replace(/^\//, "") &&
            location.hostname == this.hostname
        ) {
            var target = $(this.hash);
            target = target.length
                ? target
                : $("[name=" + this.hash.slice(1) + "]");
            if (target.length) {
                $("html, body").animate(
                    {
                        scrollTop: target.offset().top - 70,
                    },
                    1000,
                    "easeInOutExpo"
                );
                return false;
            }
        }
    });

    // Closes responsive menu when a scroll trigger link is clicked
    $(".js-scroll-trigger").click(function () {
        $(".navbar-collapse").collapse("hide");
    });

    // Activate scrollspy to add active class to navbar items on scroll
    $("body").scrollspy({
        target: "#mainNav",
        offset: 100,
    });

    // Collapse Navbar
    var navbarCollapse = function () {
        if ($("#mainNav").offset().top > 100) {
            $("#mainNav").addClass("navbar-shrink");
        } else {
            $("#mainNav").removeClass("navbar-shrink");
        }
    };
    // Collapse now if page is not at top
    navbarCollapse();
    // Collapse the navbar when page is scrolled
    $(window).scroll(navbarCollapse);
})(jQuery); // End of use strict


$(function()
{
     $('input[name="startdate"]').datepicker({
        opens: 'left'
     },
     function(start, end, label) {});

    $('input[name="enddate"]').datepicker
    ({
        opens: 'left'
    },
    function(start, end, label) {});
    date = new Date()
    $('input[name="enddate"]').val(date.getUTCMonth()+"/"+date.getUTCDate()+"/"+date.getUTCFullYear())

     $( "#slider11" ).on('change',
      function(dt)
      {
        $("span#sliderval").text($(dt.target).val())
      })

}
)

function onSubmit(dt)
{

    console.log("Submitting")
    var loglist = []

    var nsamples = $("#slider11").val()
    var start = $("#startdate").val()
    var end = $("#enddate").val()
    for(var ii=0; ii<6; ii++)
    {
        var value = $("select#log"+ii).val()
        if(value != null)
            loglist[ii-1] = value;
    }
    output = {ds_names: loglist, nsamples: nsamples}
    $.ajax
    (
    {
        url: "recent/json",
        data:JSON.stringify(output),
        content: "application/json",
        type: "post",
        success:function(dt)
        {
            console.log("submision success")
            console.log(dt)
        },
    }

    )

}

function trigger_modal(title, msg)
{
    $('#modal_title').text(title);
    $('#modal_msg').text(msg);
    $('div#error').modal();
}
