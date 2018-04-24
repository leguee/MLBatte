"use strict";

require(["comp/menubar", "comp/mainview", "comp/aboutview", "comp/dataview", "comp/vizview", "comp/ml/classificationview", "comp/ml/regressionview", "comp/ml/clusterview", "comp/ml/neuralnetworkview", "event/manager"], function (Menu, Main, About, Data, Viz, CLS, Regression, Cluster, NeuralNetwork, EventBus) {

    console.log("Nothing happend yet~ ");

    var handleMainmenu = function handleMainmenu() {
        ReactDOM.render(React.createElement(Main, null), document.getElementById('container'));
    };

    var handleAboutmenu = function handleAboutmenu() {
        ReactDOM.render(React.createElement(About, null), document.getElementById('container'));
    };

    EventBus.register("MenuEvent_About", handleAboutmenu);

    var handleDatamenu = function handleDatamenu() {
        ReactDOM.render(React.createElement(Data, null), document.getElementById('container'));
    };

    EventBus.register("MenuEvent_Data", handleDatamenu);

    var handleVizmenu = function handleVizmenu() {
        ReactDOM.render(React.createElement(Viz, null), document.getElementById('container'));
    };

    EventBus.register("MenuEvent_Visualization", handleVizmenu);

    var handleCLSmenu = function handleCLSmenu() {
        ReactDOM.render(React.createElement(CLS, null), document.getElementById('container'));
    };

    EventBus.register("MenuEvent_Classification", handleCLSmenu);

    var handleRegressionmenu = function handleRegressionmenu() {
        ReactDOM.render(React.createElement(Regression, null), document.getElementById('container'));
    };

    EventBus.register("MenuEvent_Regression", handleRegressionmenu);

    var handleClustermenu = function handleClustermenu() {
        ReactDOM.render(React.createElement(Cluster, null), document.getElementById('container'));
    };

    EventBus.register("MenuEvent_Cluster", handleClustermenu);

    var handleNeuralNetworkmenu = function handleNeuralNetworkmenu() {
        ReactDOM.render(React.createElement(NeuralNetwork, null), document.getElementById('container'));
    };

    EventBus.register("MenuEvent_NeuralNetwork", handleNeuralNetworkmenu);

    $("#home_link").click(function () {
        handleMainmenu();
    });

    var props1 = {
        name: "Data",
        id: "Data",
        items: []
    };
    var props2 = {
        name: "Analysis",
        id: "Analysis",
        items: ["Visualization", "Classification", "Cluster", "Regression", "NeuralNetwork"]
    };
    var props3 = {
        name: "Data Preprocess",
        id: "About",
        items: []
    };

    var data = [props1, props2, props3];

    ReactDOM.render(React.createElement(Menu, {
        data: data
    }), document.getElementById('menubar'));

    handleMainmenu();
});