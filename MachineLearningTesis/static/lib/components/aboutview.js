"use strict";

define([], function () {
    var AboutPage = React.createClass({
        displayName: "AboutPage",

        render: function render() {
            return React.createElement(
                "div",
                null,
                "This is the about page!"
            );
        }
    });

    return AboutPage;
});