"use strict";

define(["comp/menu"], function (Menu) {
    var MenuBar = React.createClass({
        displayName: "MenuBar",

        render: function render() {
            var items = this.props.data;
            return React.createElement(
                "ul",
                { className: "nav navbar-nav" },
                items.map(function (result) {
                    return React.createElement(Menu, { name: result.name, key: result.id, items: result.items });
                })
            );
        }
    });

    return MenuBar;
});