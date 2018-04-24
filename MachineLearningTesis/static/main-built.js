define("event/manager", [], function() {
    var e = {},
        t = {
            trigger: function(n) {
                console.log("trigger and event handle " + n);
                if (!e[n]) return;
                e[n].map(function(e) {
                    e()
                })
            },
            register: function(n, r) {
                e[n] || (e[n] = []), e[n].push(r)
            },
            unregister: function(n, r) {
                if (!e[n]) return;
                var i = e[n].indexOf(r);
                i > -1 && array.splice(i, 1)
            },
            clear: function(n) {
                if (!e[n]) return;
                e[n] = []
            }
        };
    return t
}), define("comp/menu", ["event/manager"], function(e) {
    var t = React.createClass({
            displayName: "MenuItem",
            render: function() {
                return React.createElement("li", {
                    role: "presentation"
                }, React.createElement("a", {
                    role: "menuitem",
                    className: "menu_item",
                    href: "#",
                    onClick: this.handleClick
                }, this.props.name))
            },
            handleClick: function() {
                e.trigger("MenuEvent_" + this.props.name)
            }
        }),
        n = React.createClass({
            displayName: "Menu",
            render: function() {
                var n = this.props.items;
                return n.length > 0 ? React.createElement("li", {
                    className: "dropdown"
                }, React.createElement("a", {
                    className: "dropdown-toggle",
                    "data-toggle": "dropdown"
                }, this.props.name, React.createElement("span", {
                    className: "caret"
                })), React.createElement("ul", {
                    className: "dropdown-menu",
                    role: "menu",
                    "aria-labelledby": "dropdownMenu"
                }, n.map(function(e) {
                    return React.createElement(t, {
                        name: e
                    })
                }))) : React.createElement("li", null, React.createElement("a", {
                    href: "#",
                    onClick: this.handleClick
                }, this.props.name))
            },
            handleClick: function() {
                e.trigger("MenuEvent_" + this.props.name)
            }
        });
    return n
}), define("comp/menubar", ["comp/menu"], function(e) {
    var t = React.createClass({
        displayName: "MenuBar",
        render: function() {
            var n = this.props.data;
            return React.createElement("ul", {
                className: "nav navbar-nav"
            }, n.map(function(t) {
                return React.createElement(e, {
                    name: t.name,
                    key: t.id,
                    items: t.items
                })
            }))
        }
    });
    return t
}), define("comp/mainview", [], function() {
    var e = React.createClass({
        displayName: "Main",
        render: function() {
            return React.createElement("div", null, React.createElement("img", {
                src: "resources/main.png"
            }))
        }
    });
    return e
}), define("comp/aboutview", [], function() {
    var e = React.createClass({
        displayName: "AboutPage",
        render: function() {
            return React.createElement("div", null, "Página de notas relacionada a la tesis!")
        }
    });
    return e
}), define("comp/common/collapse", [], function() {
    var e = React.createClass({
        displayName: "CollapsePanel",
        render: function() {
            var t = "#" + this.props.id,
                n = "heading_" + this.props.id;
            return React.createElement("div", {
                className: "panel panel-default"
            }, React.createElement("div", {
                className: "panel-heading",
                role: "tab",
                id: n
            }, React.createElement("h4", {
                className: "panel-title"
            }, React.createElement("a", {
                role: "button",
                "data-toggle": "collapse",
                "data-parent": "#accordion",
                href: t,
                "aria-expanded": "true",
                "aria-controls": this.props.id
            }, this.props.title))), React.createElement("div", {
                id: this.props.id,
                className: "panel-collapse collapse",
                role: "tabpanel",
                "aria-labelledby": n
            }, React.createElement("div", {
                className: "panel-body"
            }, this.props.children)))
        }
    });
    return e
}), define("comp/common/panel", [], function() {
    var e = React.createClass({
        displayName: "Panel",
        render: function() {
            var t = {};
            return t.overflow = "auto", React.createElement("div", {
                className: "panel panel-default"
            }, React.createElement("div", {
                className: "panel-heading"
            }, React.createElement("h3", {
                className: "panel-title"
            }, this.props.title)), React.createElement("div", {
                className: "panel-body",
                id: this.props.id,
                style: t
            }, this.props.children))
        }
    });
    return e
}), define("comp/dataselection", [], function() {
    var e = React.createClass({
        displayName: "DataSelectionPanel",
        render: function() {
            return React.createElement("select", {
                className: "input-medium",
                id: "selectDataSource"
            })
        },
        componentDidMount: function() {
            var t = this.props.handleSelection;
            $.get("/csvdata", function(e) {
                var n = $.parseJSON(e),
                    r = $("#selectDataSource");
                r.select2({
                    data: n,
                    tags: "true",
                    width: "256px",
                    placeholder: "Seleccione el origen de datos",
                    allowClear: !0
                }), r.val(null).trigger("change"), r.change(function() {
                    console.log("New Data Source selected : " + $(this).val()), t($(this).val())
                })
            })
        }
    });
    return e
}), define("comp/dataview", ["comp/common/collapse", "comp/common/panel", "comp/dataselection"], function(e, t, n) {
    var r = React.createClass({
            displayName: "DataUploadPanel",
            render: function() {
                return React.createElement("div", null, React.createElement("form", {
                    role: "form",
                    method: "post",
                    encType: "multipart/form-data",
                    onSubmit: this.trackFormSubmission
                }, React.createElement("div", {
                    className: "form-group"
                }, React.createElement("input", {
                    id: "input-id",
                    type: "file",
                    name: "file"
                }))))
            },
            componentDidMount: function() {
                $("#input-id").fileinput({
                    showPreview: !0,
                    uploadUrl: "/csvdata",
                    allowedFileExtensions: ["csv"]
                })
            }
        }),
        s = function(t) {
            t !== null ? o(t) : $("#dataTablePanel").empty()
        },
        o = function(t) {
            var n = {};
            $.get("/data/" + t, n, function(e) {
                var t = e.csv,
                    e = Papa.parse(t);
                console.log(e), $("#DataTablePanel").empty(), ReactDOM.render(React.createElement(f, {
                    data: e.data
                }), document.getElementById("dataTablePanel")), $("#dataTable").DataTable()
            })
        },
        u = React.createClass({
            displayName: "DataTablePanel",
            render: function() {
                return React.createElement(t, {
                    id: "dataTablePanel",
                    title: "Visualización de Datos"
                })
            },
            componentDidMount: function() {}
        }),
        a = React.createClass({
            displayName: "DataRow",
            render: function() {
                var t = this.props.data;
                return React.createElement("tr", null, t.map(function(e) {
                    return React.createElement("td", null, e)
                }))
            }
        }),
        f = React.createClass({
            displayName: "DataTable",
            render: function() {
                var t = this.props.data,
                    n = t[0];
                return t.splice(0, 1), React.createElement("table", {
                    id: "dataTable",
                    className: "table"
                }, React.createElement("thead", null, React.createElement("tr", null, n.map(function(e) {
                    return React.createElement("th", null, e)
                }))), React.createElement("tbody", null, t.map(function(e) {
                    return React.createElement(a, {
                        data: e
                    })
                })))
            },
            componentDidMount: function() {}
        }),
        l = React.createClass({
            displayName: "DataPage",
            render: function() {
                return React.createElement("div", {
                    className: "row"
                }, React.createElement("div", {
                    className: "col-md-4"
                }, React.createElement("div", {
                    className: "panel-group",
                    role: "tablist",
                    "aria-multiselectable": "true"
                }, React.createElement(e, {
                    title: "Seleccionar Datos",
                    id: "SelectDataCollapse"
                }, React.createElement(n, {
                    headerOnly: "true",
                    handleSelection: s
                })), React.createElement(e, {
                    title: "Agregar Datos",
                    id: "DataUploadCollapse"
                }, React.createElement(r, null)))), React.createElement("div", {
                    className: "col-md-8"
                }, React.createElement(u, null)))
            },
            componentDidMount: function() {
                $("#SelectDataCollapse").collapse("show")
            }
        });
    return l
}), define("data/model", [], function() {
    var e = function(t) {
        this._data = t.slice(0), this._header = this._data.splice(0, 1)[0]
    };
    return e.prototype.getSerialsByCol = function(e, t) {
        var n = this._header.indexOf(e),
            r = [];
        return n >= 0 && this._data.map(function(e) {
            var i = e[n];
            t && (i = parseFloat(i)), r.push(i)
        }), r
    }, e.prototype.getUniqueValuesByCol = function(e) {
        var t = this._header.indexOf(e),
            n = [];
        return t >= 0 && this._data.map(function(e) {
            var r = e[t];
            n.indexOf(r) < 0 && n.push(r)
        }), n
    }, e.prototype.getSerialsByFilter = function(e, t, n, r) {
        var i = this._header,
            s = i.indexOf(e),
            o = [];
        n.map(function(e) {
            var t = i.indexOf(e);
            o.push(t)
        });
        var u = [];
        return s >= 0 && this._data.map(function(e) {
            var n = e[s];
            if (n == t) {
                var i = [];
                o.map(function(t) {
                    var n = e[t];
                    r && (n = parseFloat(n)), i.push(n)
                }), u.push(i)
            }
        }), u
    }, e
}), define("viz/xc1ymn", ["data/model"], function(e) {
    var t = function() {
        this.option = {
            xAxis: [{
                type: "category",
                data: []
            }],
            yAxis: [{
                type: "value"
            }],
            series: [],
            legend: {
                data: []
            },
            toolbox: {
                show: !0,
                feature: {
                    dataView: {
                        show: !0,
                        readOnly: !1
                    },
                    magicType: {
                        show: !0,
                        type: ["line", "bar", "stack", "tiled"]
                    },
                    restore: {
                        show: !0
                    },
                    saveAsImage: {
                        show: !0
                    }
                }
            },
            tooltip: {
                trigger: "axis"
            }
        }, this.meta = {
            name: "xc1ymn",
            chartName: "none",
            binding: [{
                name: "x",
                type: "Category",
                maxFeed: 1
            }, {
                name: "y",
                type: "Measure",
                maxFeed: 10
            }]
        }
    };
    return t.prototype.buildOption = function(t, n) {
        if (!n.x || !n.y) return;
        var r = $.extend(!0, {}, this.option),
            i = new e(t),
            s = i.getSerialsByCol(n.x, !1);
        r.xAxis[0].data = s, r.legend.data = n.y;
        var o = this;
        return n.y.map(function(e) {
            var t = i.getSerialsByCol(e, !0),
                n = {};
            n.name = e, n.type = o.meta.chartName, n.data = t, o.appendOption(n), r.series.push(n)
        }), r
    }, t.prototype.appendOption = function(e) {}, t
}), define("viz/line", ["viz/xc1ymn"], function(e) {
    var t = new e;
    return t.meta.name = "line", t.meta.chartName = "line", t
}), define("viz/bar", ["viz/xc1ymn"], function(e) {
    var t = new e;
    return t.meta.name = "bar", t.meta.chartName = "bar", t
}), define("viz/area", ["viz/xc1ymn"], function(e) {
    var t = new e;
    return t.meta.name = "area", t.meta.chartName = "line", t.appendOption = function(e) {
        e.itemStyle = {
            normal: {
                areaStyle: {
                    type: "default"
                }
            }
        }
    }, t
}), define("viz/c1m1", ["data/model"], function(e) {
    var t = function() {
        this.option = {
            series: [],
            tooltip: {
                trigger: "item"
            },
            legend: {
                orient: "vertical",
                x: "left",
                data: []
            }
        }, this.meta = {
            name: "c1m1",
            chartName: "none",
            binding: [{
                name: "Category",
                type: "Category",
                maxFeed: 1
            }, {
                name: "Measure",
                type: "Measure",
                maxFeed: 1
            }]
        }
    };
    return t.prototype.buildOption = function(t, n) {
        if (!n.Category || !n.Measure) return;
        var r = $.extend(!0, {}, this.option),
            i = new e(t),
            s = i.getSerialsByCol(n.Category, !1);
        r.legend.data = s;
        var o = i.getSerialsByCol(n.Measure, !0),
            u = {};
        u.name = n.Measure, u.type = this.meta.chartName;
        var a = 0,
            f = s.length,
            l = [];
        for (; a < f; a++) {
            var c = {};
            c.name = s[a], c.value = o[a], l.push(c)
        }
        return u.data = l, this.appendOption(u), r.series.push(u), r
    }, t.prototype.appendOption = function(e) {}, t
}), define("viz/pie", ["viz/c1m1"], function(e) {
    var t = new e;
    return t.option.toolbox = {
        show: !0,
        feature: {
            dataView: {
                show: !0,
                readOnly: !1
            },
            magicType: {
                show: !0,
                type: ["pie", "funnel"],
                option: {
                    funnel: {
                        x: "25%",
                        width: "50%",
                        funnelAlign: "left",
                        max: 1548
                    }
                }
            },
            restore: {
                show: !0
            },
            saveAsImage: {
                show: !0
            }
        }
    }, t.meta.name = "pie", t.meta.chartName = "pie", t.appendOption = function(e) {
        e.radius = "55%", e.center = ["50%", "60%"]
    }, t
}), define("viz/xm1ym1cc1", ["data/model"], function(e) {
    var t = function() {
        this.option = {
            xAxis: [{
                type: "value",
                scale: !0
            }],
            yAxis: [{
                type: "value",
                scale: !0
            }],
            series: [],
            legend: {
                data: []
            },
            toolbox: {
                show: !0,
                feature: {
                    dataZoom: {
                        show: !0
                    },
                    dataView: {
                        show: !0,
                        readOnly: !1
                    },
                    restore: {
                        show: !0
                    },
                    saveAsImage: {
                        show: !0
                    }
                }
            },
            tooltip: {
                trigger: "axis",
                showDelay: 0,
                axisPointer: {
                    show: !0,
                    type: "cross",
                    lineStyle: {
                        type: "dashed",
                        width: 1
                    }
                }
            }
        }, this.meta = {
            name: "xm1ym1cc1",
            chartName: "none",
            binding: [{
                name: "x",
                type: "Measure",
                maxFeed: 1
            }, {
                name: "y",
                type: "Measure",
                maxFeed: 1
            }, {
                name: "color",
                type: "Category",
                maxFeed: 1
            }]
        }
    };
    return t.prototype.buildOption = function(t, n) {
        if (!n.x || !n.y || !n.color) return;
        var r = $.extend(!0, {}, this.option),
            i = new e(t),
            s = i.getUniqueValuesByCol(n.color);
        r.legend.data = s;
        var o = this;
        return s.map(function(e) {
            var t = {};
            t.name = e, t.type = o.meta.chartName, t.data = i.getSerialsByFilter(n.color, e, [n.x, n.y], !0), r.series.push(t)
        }), r
    }, t.prototype.appendOption = function(e) {}, t
}), define("viz/scatter", ["viz/xm1ym1cc1"], function(e) {
    var t = new e;
    return t.meta.name = "scatter", t.meta.chartName = "scatter", t
}), define("viz/treemap", ["viz/c1m1"], function(e) {
    var t = new e;
    return t.meta.name = "treemap", t.meta.chartName = "treemap", t.option.toolbox = {
        show: !0,
        feature: {
            dataView: {
                show: !0,
                readOnly: !1
            },
            restore: {
                show: !0
            },
            saveAsImage: {
                show: !0
            }
        }
    }, t.appendOption = function(e) {
        e.itemStyle = {
            normal: {
                label: {
                    show: !0,
                    formatter: "{b}"
                },
                borderWidth: 1
            },
            emphasis: {
                label: {
                    show: !0
                }
            }
        }
    }, t
}), define("viz/manager", ["viz/line", "viz/bar", "viz/area", "viz/pie", "viz/scatter", "viz/treemap"], function(e, t, n, r, i, s) {
    var o = [e.meta, t.meta, n.meta, r.meta, i.meta, s.meta],
        u = {};
    u.getVizTypes = function() {
        var e = [];
        return o.map(function(t) {
            e.push(t.name)
        }), e
    }, u.getVizBinding = function(e) {
        var t = u.getVizMeta(e);
        return t ? t.binding : undefined
    }, u.getVizMeta = function(e) {
        var t = 0,
            n = o.length;
        for (; t < n; t++)
            if (o[t].name === e) return o[t];
        return undefined
    };
    var a = function(u) {
        if (u === "line") return e;
        if (u === "bar") return t;
        if (u === "area") return n;
        if (u === "pie") return r;
        if (u === "scatter") return i;
        if (u === "treemap") return s
    };
    return u.buildOption = function(e, t, n) {
        if (!e) return;
        if (!t) return;
        if (!n) return;
        var r = a(n);
        return r ? r.buildOption(e, t) : undefined
    }, u
}), define("comp/vizselection", ["viz/manager"], function(e) {
    var t = React.createClass({
        displayName: "VizSelectionPanel",
        render: function() {
            return React.createElement("select", {
                className: "input-medium",
                id: "selectVizType"
            })
        },
        componentDidMount: function() {
            var n = this.props.handleSelection,
                r = $("#selectVizType"),
                i = e.getVizTypes();
            r.select2({
                data: i,
                tags: "true",
                width: "256px",
                placeholder: "Select a visualization type",
                allowClear: !0
            }), r.val(null).trigger("change"), r.change(function() {
                console.log("New Viz Type selected : " + $(this).val()), n($(this).val())
            })
        }
    });
    return t
}), define("comp/bindingselection", [], function() {
    var e = React.createClass({
        displayName: "BindingSelection",
        render: function() {
            return this.props.isMultiple ? React.createElement("select", {
                className: "input-medium",
                ref: "selectType",
                multiple: "multiple"
            }) : React.createElement("select", {
                className: "input-medium",
                ref: "selectType"
            })
        },
        componentDidMount: function() {
            var t = this.props.value,
                n = this.props.handleSelection,
                r = $(this.refs.selectType),
                i = this.props.name;
            r.select2({
                data: t,
                tags: "true",
                width: "256px",
                placeholder: i,
                allowClear: !0
            }), r.val(null).trigger("change"), r.change(function() {
                var e = {};
                e[i] = $(this).val(), n(e)
            })
        }
    });
    return e
}), define("comp/bindingpanel", ["comp/bindingselection"], function(e) {
    var t = {},
        n = React.createClass({
            displayName: "BindingPanel",
            render: function() {
                var n = this.props.bindings,
                    r = this.props.values,
                    i = this.handleSelect;
                return React.createElement("form", null, n.map(function(t) {
                    return React.createElement("div", {
                        className: "form-group"
                    }, React.createElement("label", null, t.name, " (", t.type, ") : "), React.createElement(e, {
                        name: t.name,
                        value: r,
                        handleSelection: i,
                        isMultiple: t.maxFeed > 1
                    }))
                }))
            },
            componentDidMount: function() {},
            handleSelect: function(n) {
                for (var r in n) t[r] = n[r];
                this.props.handleBinding(t)
            }
        });
    return n
}), define("comp/vizview", ["comp/common/collapse", "comp/common/panel", "comp/dataselection", "comp/vizselection", "comp/bindingpanel", "viz/manager"], function(e, t, n, r, i, s) {
    var o = undefined,
        u = undefined,
        a = undefined,
        f = function(t) {
            t !== null ? c(t) : o = undefined
        },
        l = function(t) {
            var n = undefined;
            t !== null ? (u = t, n = s.getVizBinding(t)) : (u = undefined, n = undefined), h(n), d()
        },
        c = function(t) {
            $.get("/data/" + t, function(e) {
                var t = e.csv,
                    e = Papa.parse(t);
                o = e.data
            })
        },
        h = function(t) {
            $("#vizBindingPanel").empty();
            if (t) {
                var n = {};
                n.bindings = t, n.values = o[0], n.handleBinding = p, ReactDOM.render(React.createElement(i, n), document.getElementById("vizBindingPanel"))
            }
        },
        p = function(t) {
            a = t, d()
        },
        d = function() {
            var t = s.buildOption(o, a, u);
            if (t) {
                var n = echarts.init(document.getElementById("chartPanel"));
                n.setOption(t)
            }
        },
        v = React.createClass({
            displayName: "VizPage",
            render: function() {
                var s = {};
                return s.height = "400px", s.width = "100%", React.createElement("div", {
                    className: "row"
                }, React.createElement("div", {
                    className: "col-md-4"
                }, React.createElement("div", {
                    className: "panel-group",
                    role: "tablist",
                    "aria-multiselectable": "true"
                }, React.createElement(e, {
                    title: "Seleccionar Datos",
                    id: "SelectDataCollapse"
                }, React.createElement(n, {
                    handleSelection: f
                })), React.createElement(e, {
                    title: "Seleccione el tipo de visualización",
                    id: "SelectVizCollapse"
                }, React.createElement(r, {
                    handleSelection: l
                })), React.createElement(e, {
                    title: "Select Data Binding Option",
                    id: "SelectBindingCollapse"
                }, React.createElement("div", {
                    id: "vizBindingPanel"
                })))), React.createElement("div", {
                    className: "col-md-8"
                }, React.createElement(t, {
                    id: "VizMain",
                    title: "Visualization"
                }, React.createElement("div", {
                    id: "chartPanel",
                    style: s
                }))))
            },
            componentDidMount: function() {
                $("#SelectDataCollapse").collapse("show"), $("#SelectVizCollapse").collapse("show"), $("#SelectBindingCollapse").collapse("show")
            }
        });
    return v
}), define("comp/common/tab", [], function() {
    var e = React.createClass({
        displayName: "Tab",
        render: function() {
            var t = {};
            t.overflow = "auto";
            var n = this.props.data;
            return React.createElement("div", {
                ref: "Tab"
            }, React.createElement("ul", {
                className: "nav nav-tabs",
                role: "tablist"
            }, n.map(function(e) {
                var t = "#" + e.name;
                return React.createElement("li", {
                    role: "presentation"
                }, React.createElement("a", {
                    href: t,
                    "aria-controls": e.name,
                    role: "tab",
                    "data-toggle": "tab"
                }, e.title))
            })), React.createElement("div", {
                className: "tab-content"
            }, n.map(function(e) {
                return React.createElement("div", {
                    role: "tabpanel",
                    className: "tab-pane",
                    id: e.name
                })
            })))
        },
        componentDidMount: function() {
            var t = this.refs.Tab;
            $(t).find("a:first").tab("show")
        }
    });
    return e
}), define("ml/viz/clsviz", [], function() {
    var e = {
            top: 50,
            right: 30,
            bottom: 50,
            left: 50
        },
        t = function(n) {
            this._trainData = n.data, this._features = n.features, this._predictData = n.predict, this._predictScale = n.scale, this._rootContainerId = n.containerId, this._width = n.size.width - e.left - e.right, this._height = n.size.height - e.top - e.bottom, this._xScale = d3.scale.linear().range([0, this._width]), this._yScale = d3.scale.linear().range([this._height, 0]), this._color = d3.scale.category10(), this._xAxis = d3.svg.axis().scale(this._xScale).orient("bottom"), this._yAxis = d3.svg.axis().scale(this._yScale).orient("left"), this._svg = d3.select("#" + this._rootContainerId).append("svg").attr("width", this._width + e.left + e.right).attr("height", this._height + e.top + e.bottom).append("g").attr("transform", "translate(" + e.left + "," + e.top + ")")
        };
    return t.prototype.render = function() {
        var e = this._xScale,
            t = this._yScale,
            n = this._color,
            r = d3.extent(this._trainData, function(e) {
                return e.x
            }),
            i = d3.extent(this._trainData, function(e) {
                return e.y
            });
        e.domain(r).nice(), t.domain(i).nice(), this._svg.append("g").attr("class", "x axis").attr("transform", "translate(0," + (this._height + 10) + ")").call(this._xAxis).append("text").attr("class", "label").attr("x", this._width).attr("y", -6).style("text-anchor", "end").text(this._features[0]), this._svg.append("g").attr("class", "y axis").attr("transform", "translate(-10,0)").call(this._yAxis).append("text").attr("class", "label").attr("transform", "rotate(-90)").attr("y", 6).attr("dy", ".71em").style("text-anchor", "end").text(this._features[1]), this._svg.append("g").attr("class", "train").selectAll(".dot").data(this._trainData).enter().append("circle").attr("class", "dot").attr("r", 3.5).attr("cx", function(t) {
            return e(t.x)
        }).attr("cy", function(e) {
            return t(e.y)
        }).style("fill", function(e) {
            return n(e.label)
        });
        var s = (e(r[1]) - e(r[0])) / this._predictScale,
            o = (t(i[0]) - t(i[1])) / this._predictScale;
        this._svg.append("g").attr("class", "predict").selectAll(".area").data(this._predictData).enter().append("rect").attr("class", "area").attr("x", function(t) {
            return e(t.x)
        }).attr("y", function(e) {
            return t(e.y)
        }).attr("width", s).attr("height", o).style("fill", function(e) {
            return n(e.label)
        }).style("fill-opacity", .3);
        var u = this._svg.selectAll(".legend").data(n.domain()).enter().append("g").attr("class", "legend").attr("transform", function(e, t) {
            return "translate(" + (t * 100 - 400) + ", -30)"
        });
        u.append("rect").attr("x", this._width - 18).attr("width", 18).attr("height", 18).style("fill", n), u.append("text").attr("x", this._width - 24).attr("y", 9).attr("dy", ".35em").style("text-anchor", "end").text(function(e) {
            return e
        })
    }, t
}), define("ml/manager", [], function() {
    var e = {},
        t = [{
            name: "Label",
            type: "Category",
            maxFeed: 1
        }, {
            name: "Features",
            type: "Measure",
            maxFeed: 10
        }],
        n = [{
            name: "Target",
            type: "Measure",
            maxFeed: 1
        }, {
            name: "Features",
            type: "Measure",
            maxFeed: 10
        }],
        r = [{
            name: "Features",
            type: "Measure",
            maxFeed: 10
        }],
        i = [{
            name: "Label",
            type: "Category",
            maxFeed: 1
        }, {
            name: "Features",
            type: "Measure",
            maxFeed: 10
        }];
    return e.getBinding = function(e) {
        return e === "Classification" ? t : e === "Regression" ? n : e === "Cluster" ? r : e == "NeuralNetwork" ? i : undefined
    }, e
}), define("comp/ml/mlview", ["comp/common/collapse", "comp/common/panel", "comp/dataselection", "comp/bindingpanel", "comp/common/tab", "ml/viz/clsviz", "ml/manager"], function(e, t, n, r, i, s, o) {
    var u = undefined,
        a = undefined,
        f = "Predict Result",
        l = "/data/",
        c = undefined,
        h = undefined,
        p = undefined,
        d = undefined,
        v = undefined,
        m = undefined,
        g = undefined,
        y = undefined,
        b = undefined,
        w = undefined,
        E = undefined,
        S = undefined,
        x = [{
            name: "viz",
            title: "Visualizar"
        },{
            name: "Metric",
            title: "Metricas"
        }, {
            name: "predict",
            title: "Predecir"
        }],
        T = function() {
            c = "/ml/" + a + "/create", h = "/ml/" + a + "/train", p = "/ml/" + a + "/predictViz", d = "/ml/" + a + "/predict", v = "/mlmodel/list/" + u
        },
        N = function(t) {
            $("#SelectBindingCollapse").collapse("show"), $("#mlBindingPanel").empty(), b = t, t ? C(t) : y = undefined
        },
        C = function(t) {
            var n = {};
            n.headerOnly = !0, $.get(l + t, n, function(e) {
                var t = e.csv,
                    e = Papa.parse(t);
                y = e.data;
                var n = {};
                n.bindings = o.getBinding(u), n.values = y[0], n.handleBinding = k, ReactDOM.render(React.createElement(r, n), document.getElementById("mlBindingPanel"))
            })
        },
        k = function(t) {
            E = t, $("#TrainCollapse").collapse("show")
        },
        L = function() {
            console.log("strat to train the whole dataset~");
            var t = {};
            t.type = w, $.get(c, t, function(e) {
                if (e.result === "Success") {
                    S = e.model;
                    var t = {};
                    t.id = S, t.data = b, m(t, E), $.get(h, t, function(e) {
                        e.result === "Success" ? (A(), O(e)) : alert("Failed to train the model : " + w);
                        alert("Entro despues de train en el servidor model : " + w);
                    })
                } else alert("Failed to create the model : " + w)
            })
        },
        A = function() {
            var t = {};
            t.cols = E.Features.slice(0), ReactDOM.render(React.createElement(_, t), document.getElementById("predict"));
            var n = {};
            n.id = "vizpanel", ReactDOM.render(React.createElement(D, n), document.getElementById("viz"));
            var j = {};
            j.id = "metricpanel", ReactDOM.render(React.createElement(Z, j), document.getElementById("Metric"))
        },
        O = function(e) {

             var n = $.parseJSON(e.metric.replace(/\'/g, '"'));
             console.log(n);

             /*Tratar de cargarlo en una tabla con td, tr  como el ejemplo de las predicciones*/

             var json = [{id: 1, name: 'Carlitos', age: 30}, {id: 2, name: 'Miguel', age: 32}, {id: 3, name: 'Amanda', age: 35}];
            createTableFromJson('metricpanel', n , true, 'tableStyle');

            function createTableFromJson(containerID, json, withHeader, cssName)
            {
                var table = $('<table>');
                table.addClass(cssName);
                $.each(json, function(k, v) {
                  var o = json[k];
                  var row = $('<tr>');
                  $.each(o, function(a, b) {
                    if (withHeader && k == 0)
                    {
                      var head = $('<th>');
                      head.text(a);
                      table.append(head);
                    }
                    var cell = $('<td>');
                    cell.text(b);
                    row.append(cell);
                  });
                  table.append(row);
                });
                $('#' + containerID).append(table);
            }


            var t = {};
            t.id = S, t.scale = 30, $.get(p, t, function(e) {
                if (e.result === "Success") {
                    var n = $.parseJSON(e.predict.replace(/\'/g, '"'));
                    console.log(n);
                    var r = {};
                    r.size = {}, r.size.width = $("#vizpanel").width(), r.size.height = $("#vizpanel").height(), r.features = E.Features, r.containerId = "vizpanel", r.data = n.data, r.predict = n.predict, r.scale = t.scale, $("#vizpanel").empty(), g(r, E)
                }
            })
        },
        M = React.createClass({
            displayName: "PredictRow",
            render: function() {
                var t = this.props.data;
                return React.createElement("tr", {
                    ref: "predictRow"
                }, t.map(function(e) {
                    return e !== f ? React.createElement("td", null, React.createElement("input", {
                        placeholder: e
                    })) : React.createElement("td", {
                        key: "resultCell"
                    }, React.createElement("div", {
                        ref: "resultCell"
                    }))
                }))
            },
            componentDidMount: function() {
                var t = this.refs.predictRow,
                    n = this.refs.resultCell;
                $(t).focusout(function() {
                    var e = [],
                        t = !0;
                    $(this).find("input").each(function(n, r) {
                        var i = $(this).val();
                        i.length > 0 ? e.push(parseFloat(i)) : t = !1
                    });
                    if (t) {
                        var r = {};
                        r.id = S, r.data = JSON.stringify([e]), $.get(d, r, function(e) {
                            if (e.result === "Success") {
                                var t = $.parseJSON(e.predict.replace(/\'/g, '"'));
                                $(n).text(t[0])
                            } else alert("failed to predict the result!")
                        })
                    }
                })
            }
        }),
        _ = React.createClass({
            displayName: "PredictDataTable",
            render: function() {
                var t = this.props.cols;
                return t.push(f), React.createElement("table", {
                    id: "predictTable",
                    className: "table"
                }, React.createElement("thead", null, React.createElement("tr", null, t.map(function(e) {
                    return React.createElement("th", null, e)
                }))), React.createElement("tbody", null, React.createElement(M, {
                    data: t
                })))
            },
            componentDidMount: function() {}
        }),
        D = React.createClass({
            displayName: "PredictVizPanel",
            render: function() {
                var t = {};
                return t.width = "100%", t.height = "400px", React.createElement("div", {
                    id: this.props.id,
                    style: t
                })
            },
            componentDidMount: function() {}
        }),
        Z = React.createClass({
            displayName: "MetricPanel",
            render: function() {
                var t = {};
                // TODO tratar de poner acá la carga de los diferentes divs
                return t.width = "100%", t.height = "400px", React.createElement("div", {
                    id: this.props.id,
                    style: t
                })
            },
            componentDidMount: function() {}
        }),
        P = React.createClass({
            displayName: "MLPage",
            render: function() {
                return u = this.props.modelName, a = this.props.modelShortName, m = this.props.buildTrainParameterHandler, g = this.props.renderPredictVizHandler, T(), React.createElement("div", {
                    className: "row"
                }, React.createElement("div", {
                    className: "col-md-4"
                }, React.createElement("div", {
                    className: "panel-group",
                    role: "tablist",
                    "aria-multiselectable": "true"
                }, React.createElement(e, {
                    title: "Seleccione el modelo",
                    id: "SelectModelCollapse"
                }, React.createElement("select", {
                    className: "input-medium",
                    id: "selectModelType"
                })), React.createElement(e, {
                    title: "Seleccione los datos",
                    id: "SelectDataCollapse"
                }, React.createElement(n, {
                    handleSelection: N
                })), React.createElement(e, {
                    title: "Seleccione Propiedades",
                    id: "SelectBindingCollapse"
                }, React.createElement("div", {
                    id: "mlBindingPanel"
                })), React.createElement(e, {
                    title: "Modelo de entrenamiento",
                    id: "TrainCollapse"
                }, React.createElement("div", null, React.createElement("button", {
                    className: "btn btn-default btn-xs",
                    onClick: L
                }, "Train"))))), React.createElement("div", {
                    className: "col-md-8"
                }, React.createElement(t, {
                    id: "MLMain",
                    title: "Vista Predicción"
                }, React.createElement(i, {
                    data: x
                }))))
            },
            componentDidMount: function() {
                $("#SelectModelCollapse").collapse("show"), $.get(v, function(e) {
                    var e = $.parseJSON(e),
                        t = $("#selectModelType");
                    t.select2({
                        data: e,
                        tags: "true",
                        width: "256px",
                        placeholder: "Seleccione un modelo",
                        allowClear: !0
                    }), t.val(null).trigger("change"), t.change(function() {
                        $("#SelectDataCollapse").collapse("show"), w = $(this).val()
                    })
                })
            }
        });
    return P
}), define("comp/ml/classificationview", ["comp/ml/mlview", "ml/viz/clsviz"], function(e, t) {
    var n = "Classification",
        r = "cls",
        i = function(t, n) {
            t.label = n.Label, t.features = n.Features.join()
        },
        s = function(n, r) {
            var i = new t(n);
            i.render()
        },
        o = React.createClass({
            displayName: "ClassificationPage",
            render: function() {
                return React.createElement(e, {
                    modelName: n,
                    modelShortName: r,
                    buildTrainParameterHandler: i,
                    renderPredictVizHandler: s
                })
            },
            componentDidMount: function() {}
        });
    return o
}), define("ml/viz/regressionviz", [], function() {
    var e = {
            top: 50,
            right: 30,
            bottom: 50,
            left: 50
        },
        t = function(n) {
            this._trainData = n.data, this._features = n.features, this._target = n.target, this._predictData = n.predict, this._predictScale = n.scale, this._rootContainerId = n.containerId, this._width = n.size.width - e.left - e.right, this._height = n.size.height - e.top - e.bottom, this._xScale = d3.scale.linear().range([0, this._width]), this._yScale = d3.scale.linear().range([this._height, 0]), this._color = d3.scale.category10(), this._xAxis = d3.svg.axis().scale(this._xScale).orient("bottom"), this._yAxis = d3.svg.axis().scale(this._yScale).orient("left"), this._svg = d3.select("#" + this._rootContainerId).append("svg").attr("width", this._width + e.left + e.right).attr("height", this._height + e.top + e.bottom).append("g").attr("transform", "translate(" + e.left + "," + e.top + ")")
        };
    return t.prototype.render = function() {
        var e = this._xScale,
            t = this._yScale,
            n = this._color,
            r = d3.extent(this._trainData, function(e) {
                return e.x
            }),
            i = d3.extent(this._trainData, function(e) {
                return e.y
            });
        e.domain(r).nice(), t.domain(i).nice(), this._svg.append("g").attr("class", "x axis").attr("transform", "translate(0," + (this._height + 10) + ")").call(this._xAxis).append("text").attr("class", "label").attr("x", this._width).attr("y", -6).style("text-anchor", "end").text(this._features[0]), this._svg.append("g").attr("class", "y axis").attr("transform", "translate(-10,0)").call(this._yAxis).append("text").attr("class", "label").attr("transform", "rotate(-90)").attr("y", 6).attr("dy", ".71em").style("text-anchor", "end").text(this._target), this._svg.selectAll(".dot").data(this._trainData).enter().append("circle").attr("class", "dot").attr("r", 3.5).attr("cx", function(t) {
            return e(t.x)
        }).attr("cy", function(e) {
            return t(e.y)
        }).style("fill", function(e) {
            return n("data")
        });
        var s = (e(r[1]) - e(r[0])) / this._predictScale,
            o = (t(i[0]) - t(i[1])) / this._predictScale,
            u = d3.svg.line().x(function(t) {
                return e(t.x)
            }).y(function(e) {
                return t(e.y)
            }).interpolate("basis");
        this._svg.append("path").attr("d", u(this._predictData)).style("stroke", n("regression")).style("stroke-width", 3).style("fill", "None")
    }, t
}), define("comp/ml/regressionview", ["comp/ml/mlview", "ml/viz/regressionviz"], function(e, t) {
    var n = "Regression",
        r = "regression",
        i = function(t, n) {
            t.target = n.Target, t.train = n.Features.join()
        },
        s = function(n, r) {
            n.target = r.Target;
            var i = new t(n);
            i.render()
        },
        o = React.createClass({
            displayName: "RegressionPage",
            render: function() {
                return React.createElement(e, {
                    modelName: n,
                    modelShortName: r,
                    buildTrainParameterHandler: i,
                    renderPredictVizHandler: s
                })
            },
            componentDidMount: function() {}
        });
    return o
}), define("comp/ml/clusterview", ["comp/ml/mlview", "ml/viz/clsviz"], function(e, t) {
    var n = "Cluster",
        r = "cluster",
        i = function(t, n) {
            t.train = n.Features.join()
        },
        s = function(n, r) {
            n.target = r.Target;
            var i = new t(n);
            i.render()
        },
        o = React.createClass({
            displayName: "ClusterPage",
            render: function() {
                return React.createElement(e, {
                    modelName: n,
                    modelShortName: r,
                    buildTrainParameterHandler: i,
                    renderPredictVizHandler: s
                })
            },
            componentDidMount: function() {}
        });
    return o
}), define("comp/ml/neuralnetworkview", ["comp/ml/mlview", "ml/viz/clsviz"], function(e, t) {
    var n = "NeuralNetwork",
        r = "neuralnetwork",
        i = function(t, n) {
            t.label = n.Label, t.features = n.Features.join()
        },
        s = function(n, r) {
            n.target = r.Target;
            var i = new t(n);
            i.render()
        },
        o = React.createClass({
            displayName: "NeuralNetworkPage",
            render: function() {
                return React.createElement(e, {
                    modelName: n,
                    modelShortName: r,
                    buildTrainParameterHandler: i,
                    renderPredictVizHandler: s
                })
            },
            componentDidMount: function() {}
        });
    return o
}), require(["comp/menubar", "comp/mainview", "comp/aboutview", "comp/dataview", "comp/vizview", "comp/ml/classificationview", "comp/ml/regressionview", "comp/ml/clusterview", "comp/ml/neuralnetworkview", "event/manager"], function(e, t, n, r, i, s, o, u, a, f) {
    console.log("Nothing happend yet~ ");
    var l = function() {
            ReactDOM.render(React.createElement(t, null), document.getElementById("container"))
        },
        c = function() {
            ReactDOM.render(React.createElement(n, null), document.getElementById("container"))
        };
    f.register("MenuEvent_About", c);
    var h = function() {
        ReactDOM.render(React.createElement(r, null), document.getElementById("container"))
    };
    f.register("MenuEvent_Data", h);
    var p = function() {
        ReactDOM.render(React.createElement(i, null), document.getElementById("container"))
    };
    f.register("MenuEvent_Visualization", p);
    var d = function() {
        ReactDOM.render(React.createElement(s, null), document.getElementById("container"))
    };
    f.register("MenuEvent_Classification", d);
    var v = function() {
        ReactDOM.render(React.createElement(o, null), document.getElementById("container"))
    };
    f.register("MenuEvent_Regression", v);
    var m = function() {
        ReactDOM.render(React.createElement(u, null), document.getElementById("container"))
    };
    f.register("MenuEvent_Cluster", m);
    var g = function() {
        ReactDOM.render(React.createElement(a, null), document.getElementById("container"))
    };
    f.register("MenuEvent_NeuralNetwork", g), $("#home_link").click(function() {
        l()
    });
    var y = {
            name: "Data",
            id: "Data",
            items: []
        },
        b = {
            name: "Algoritmos ML",
            id: "Analysis",
            items: ["Visualization", "Classification", "Cluster", "Regression", "NeuralNetwork"]
        },
        w = {
            name: "About",
            id: "About",
            items: []
        },
        z = {
            name: "Metricas",
            id: "Metricas",
            items: []
        },
        E = [y, b, z, w];
    ReactDOM.render(React.createElement(e, {
        data: E
    }), document.getElementById("menubar")), l()
}), define("app", function() {});