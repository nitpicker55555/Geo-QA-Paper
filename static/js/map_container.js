function updateMapDataAndFitBounds(map,geojson,target_labels=[]){
    function createCustomClusterIcon(cluster) {
        var childMarkers = cluster.getAllChildMarkers();
        var color = childMarkers[0].options.color; // 读取自定义颜色属性
        return L.divIcon({
            html: '<div class="custom-cluster-icon" style="background-color:' + color + ';">' + cluster.getChildCount() + '</div>',
            className: 'custom-cluster',
            iconSize: [40, 40]
        });
    }
    // 示例列表
    var markers = L.markerClusterGroup({
        iconCreateFunction: createCustomClusterIcon
    });

    console.log(target_labels)
    if (window.geoJsonLayer) {
        window.geoJsonLayer.clearLayers(); // 移除之前的图层数据
    } else {
        window.geoJsonLayer = L.geoJSON().addTo(map); // 初始化geoJsonLayer

    }
    removeAllControls()
    function removeAllControls() {
        allControls.forEach(function(control) {
            map.removeControl(control);
        });
        allControls = []; // 清空数组
    }

    function getRandomColor(label) {
        const label_belong =label.split("_").slice(0, 2).join("_");
        // const lastIndex = label.lastIndexOf("_"); // 找到最后一个下划线的索引
        // const label_belong = label.substring(0, lastIndex);
        if (label_belong in color_dict){
            return color_dict[label_belong]
        }
        else
        {
            const color =picker.getNextElement()
            // console.log(color,label_belong)
            color_dict[label_belong]=color
            return color;
        }


    }
    function createSequentialPicker(array) {
        if (!Array.isArray(array) || array.length === 0) {
            throw new Error("Array must be a non-empty array.");
        }
        function getNextElement() {
            // 获取当前索引的元素
            const element = array[currentIndex];

            // 更新索引，如果到达数组末尾则重置为0
            currentIndex = (currentIndex + 1) % array.length;

            // 记录已经抽取过的元素
            usedElements.push(element);

            return element;
        }

        function getUsedElements() {
            return usedElements;
        }

        return {
            getNextElement,
            getUsedElements
        };
    }
    const picker = createSequentialPicker(color_list);
    function create_container_list(labels_ori,div){
        var blocks = {};
        let labels=[]
        let colors_btn={}

        const keys = Object.keys(labels_ori);

        // 获取第一个元素的键和值
        const firstKey = keys[0];
        const firstValue = labels_ori[firstKey];

        // 创建一个新的对象，并将第一个元素添加进去
        const sortedObj = { [firstKey]: firstValue };

        // 移除第一个元素的键
        keys.shift();

        // 对剩余的键进行排序
        keys.sort();

        // 将排序后的元素依次加入新对象
        keys.forEach(key => {
            sortedObj[key] = labels_ori[key];
        });
        // console.log(labels_ori)

        // console.log(sortedObj)
        for (var label in sortedObj) {
            (function(label) {
                // Check if the label exists in labels_ori before accessing its properties
                if (!labels_ori[label]) {
                    return; // Skip this label if it doesn't exist
                }
                let color=labels_ori[label]['color']
                colors_btn[label]=color
                let layer=labels_ori[label]['layer']
                if (!labels[label]) {

                    // 获取第一个下划线前面的内容作为block的key
                    var blockKey = label.includes('_') ? "💾 "+label.split('_')[0] : '🗺️ Region';


                    if (!blocks[blockKey]) {
                        // 如果blockKey不存在，创建一个新的block
                        blocks[blockKey] = {
                            button: L.DomUtil.create('button', 'toggle-block-btn', div),
                            labels: {}
                        };
                        blocks[blockKey].button.style.backgroundColor = 'grey';
                        blocks[blockKey].button.textContent = blockKey;
                        blocks[blockKey].button.classList.add('block-button');
                        blocks[blockKey].button.onclick = function() {
                            var blockBtn = blocks[blockKey].button;
                            var labelsInBlock = blocks[blockKey].labels;
                            var allActive = true;

                            for (var lbl in labelsInBlock) {
                                if (!map.hasLayer(labelsInBlock[lbl].layers[0])) {
                                    allActive = false;
                                    break;
                                }
                            }

                            if (allActive) {
                                for ( lbl in labelsInBlock) {
                                    labelsInBlock[lbl].layers.forEach(function(layer) {
                                        map.removeLayer(layer);
                                    });
                                    labelsInBlock[lbl].button.style.backgroundColor = 'black';
                                }
                            } else {
                                for ( lbl in labelsInBlock) {
                                    labelsInBlock[lbl].layers.forEach(function(layer) {
                                        map.addLayer(layer);
                                    });
                                    // console.log(lbl)
                                    // console.log(labelsInBlock[lbl])
                                    // console.log(colors[lbl])
                                    labelsInBlock[lbl].button.style.backgroundColor = colors_btn[lbl];
                                }
                            }
                        };
                    }

                    // 创建一个新的label按钮
                    labels[label] = {
                        button: L.DomUtil.create('button', 'toggle-btn', div),
                        layers: []
                    };

                    labels[label].button.style.backgroundColor = color;
                    // console.log(labels[label])

                    labels[label].button.textContent =
                        labels[label].button.textContent = target_labels.includes(label) ? label + " (target)" : label+" ("+labels_ori[label]['layer'].length+")"
                    labels[label].button.onclick = function() {
                        var btn = labels[label].button;
                        var layers = labels[label].layers;
                        if (map.hasLayer(layers[0])) {
                            layers.forEach(function(layer) {
                                map.removeLayer(layer);
                            });
                            btn.style.backgroundColor = 'black';
                        } else {
                            layers.forEach(function(layer) {
                                map.addLayer(layer);
                            });

                            btn.style.backgroundColor = color;
                        }
                    };

                    // 将当前label添加到对应的block中
                    blocks[blockKey].labels[label] = labels[label];
                    // index++;
                }
                labels[label].layers=layer;
            })(label);
        }

    }
    var controls = L.control({position: 'topleft'});
    function push_obj(obj1, obj2) {
        Object.assign(obj1, obj2);
    }

    controls.onAdd = function(map) {
        var div = L.DomUtil.create('div', 'leaflet-control-custom');
        var labels = {}; // to keep track of unique labels


        for (var key in geojson) {
            (function(key) {
                var parts = key.split('_');
                const lastIndex = key.lastIndexOf("_"); // 找到最后一个下划线的索引

                var label = parts.length > 2 ? parts.slice(0, 2).join('_') : key;// use the second element as label or key if not present
                var label_belong = key.substring(0, lastIndex); // use the second element as label or key if not present

                var color = getRandomColor(key);
                var feature = geojson[key];
                var layerGroup = L.featureGroup();
                var layerOptions = {
                    onEachFeature: function(feature, layer) {
                        layer.bindPopup(label_belong);
                        if (target_labels.includes(label)){
                            if (feature.type === 'Point') {
                                latlng = L.latLng(feature.coordinates[1], feature.coordinates[0]);
                            } else {
                                latlng = L.latLngBounds(layer.getBounds()).getCenter();
                            }

                            const layer_target=L.marker(latlng, {
                                icon: L.divIcon({
                                    className: 'custom-div-icon',
                                    html:  `
            <div style="
                position: relative;
                width: 20px;
                height: 30px;
                background-color: transparent;">
                <div style="
                    width: 20px;
                    height: 20px;
                    background-color: ${color};
                    border: 2px solid white;
                    border-radius: 50%;
                    position: absolute;
                    top: 0;
                    left: 50%;
                    transform: translateX(-50%);
                    box-shadow: 0 0 0 2px ${color};">
                </div>

            </div>
        `,
                                    iconSize: [20, 20]
                                }),
                                color: color // 将颜色作为自定义属性附加到标记上
                            })
                            markers.addLayer(layer_target)
                            layerGroup.addLayer(markers)
                            layer_target.bindPopup(label_belong);

                        }

                    }
                };




                if (feature.type === "Point") {
                    push_obj(layerOptions, {

                        pointToLayer: function (feature, latlng) {
                            return L.circleMarker(latlng, {
                                radius: 5, // 调整半径大小以改变点的大小
                                fillColor: color, //
                                color:color, //
                                weight: 1,
                                opacity: 1,
                                fillOpacity: 0.8
                            });
                        },

                    });
                }else {
                    push_obj(layerOptions, {

                        style: function(feature) {
                            return {color: color, weight: 2};
                        }
                    })
                }
                var layer = L.geoJSON(feature, layerOptions)
                layerGroup.addLayer(layer)
                layerGroup.addTo(window.geoJsonLayer);
                if     ( !labels[label])  {
                    labels[label]={
                        color: color,
                        layer: [layerGroup]
                    }

                }else {
                    labels[label]['layer'].push(layerGroup)
                };







            })(key);
        }
        // console.log(labels)
        // console.log(labels)
        create_container_list(labels,div)
        return div;
    };
    controls.addTo(map);
    allControls.push(controls)

    // Create custom control for the toggle button
    var toggleControl = L.control({position: 'topleft'});
    toggleControl.onAdd = function(map) {
        var button = L.DomUtil.create('button', 'leaflet-control-toggle');
        // var rect = button.getBoundingClientRect();
        // console.log('控件的位置:', rect);
        // console.log(button.style.top)
        button.innerHTML = '⬆️';

        button.onclick = function() {
            var controlsDiv = document.querySelector('.leaflet-control-custom');
            const windowHeight = window.innerHeight;
            const controlsHeight = controlsDiv.offsetHeight;
            if (controlsDiv.style.top === '50px'||controlsDiv.style.top === '') {
                controlsDiv.style.top = -controlsHeight+'px';// Adjust based on your actual controls height
                button.innerHTML = '⬇️';
                // console.log(controlsHeight)
            } else {
                controlsDiv.style.top = '50px';
                button.innerHTML = '⬆️';
            }
        };
        return button;
    };
    toggleControl.addTo(map);
    allControls.push(toggleControl)

    // var controlInstance = control.addTo(map);

    // Toggle button for collapsing control panel
    // var toggle = document.getElementById('toggle');


    var expandControl = L.control({position: 'bottomleft'});

    expandControl.onAdd = function (map) {
        var div = L.DomUtil.create('div', 'leaflet-control-expand');
        div.innerHTML = '<button id="expandBtn" style="background: #fff; border: none; border-radius: 4px; box-shadow: 0 1px 5px rgba(0,0,0,0.3); padding: 8px 15px; cursor: pointer;">Expand</button>';
        return div;
    };


    expandControl.addTo(map);
    allControls.push(expandControl)
    var mapElement = document.getElementById('map-container');
    var expandBtn = document.getElementById('expandBtn');

    expandBtn.onclick = function() {
        if (mapElement.classList.contains('fullscreen')) {
            mapElement.classList.remove('fullscreen');
            expandBtn.textContent = 'Expand';
            updateControlHeight();


        } else {
            mapElement.classList.add('fullscreen');
            expandBtn.textContent = 'Shrink';
            updateControlHeight();

        }
        map.invalidateSize(); // Leaflet needs to adjust to new container size
    };

    // Listen for escape key to shrink map if in fullscreen
    document.addEventListener('keydown', function(event) {
        if (event.key === "Escape" && mapElement.classList.contains('fullscreen')) {
            mapElement.classList.remove('fullscreen');
            expandBtn.textContent = 'Expand';
            const map_height = mapElement.clientHeight;

            control_custom.style.maxHeight = map_height / 2;
            map.invalidateSize(); // Adjust map after resizing
        }
    });
    try {
        map.fitBounds(window.geoJsonLayer.getBounds());
    } catch (error) {
        console.log('Error fitting bounds:', window.geoJsonLayer.getBounds());
    }
    updateControlHeight()

}
