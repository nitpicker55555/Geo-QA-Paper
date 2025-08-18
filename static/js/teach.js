function waitForTrue(getVariableValue, checkInterval = 500, initialDelay = 2000) {
    return new Promise((resolve) => {
        setTimeout(() => {
            const intervalId = setInterval(() => {
                if (!getVariableValue()) { // 检查getVariableValue()的返回值是否为false
                    clearInterval(intervalId);
                    resolve(true);
                }
                // console.log(getVariableValue())
            }, checkInterval);
        }, initialDelay);
    });
}
function getVariableValue() {
    // console.log(process_status)
    return process_status;
}

function change_overlay_zindex(mode){


    $('.overlay').css('z-index',mode);


}
var stop_sign=false
function teach_assistant(tips){
    // console.log(tips)
    document.getElementById('stop-tips').style.display = 'inline-block';



        let currentTipIndex = 0;

        const overlay = document.getElementById('overlay');
        const tooltip = document.getElementById('tooltip');
        const tooltipText = document.getElementById('tooltip-text');
        var tooltipButton = document.getElementById('tooltip-button');
        const stopTipsButton = document.getElementById('stop-tips');
        const center_tooltip=document.getElementById('centeredTooltip');


        stopTipsButton.addEventListener('click', function () {
            stop_sign = true;
            tooltip.classList.remove('visible');
            center_tooltip.classList.remove('visible');
            overlay.classList.remove('visible');
            document.querySelectorAll('.highlight').forEach(function(element) {
                element.classList.remove('highlight');
            });
            // element.classList.remove('highlight');


            setTimeout(() => {
                tooltip.classList.remove('top', 'bottom', 'center');
                tooltip.style.display = 'none';
                center_tooltip.style.display = 'none';
            }, 500); // Wait for the fade-out animation to complete
            currentTipIndex=tips.length
            // showTip(tips.length-1)
            // tooltipButton = document.getElementById('tooltip-button');
            document.getElementById('stop-tips').style.display = 'none';
        });

        function checkElementExists(xpath) {
            return new Promise(resolve => {
                const interval = setInterval(() => {
                    const element = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                    if (element) {
                        clearInterval(interval);
                        resolve(element);
                    }

                    console.log('wait for elements',xpath)
                }, 500); // Check every 100ms
            });
        }

        async function showTip(index) {
            const tip = tips[index];
            // console.log(tip,index)
            if (!tip) return;

            const element = await checkElementExists(tip.xpath);
            if (!element) return;

            // Add highlight class to element
            element.classList.add('highlight');
            // if ( tip.text.includes('cluster')){
            //     let addtional_html=get_innerhtml(tip.xpath)
            //     console.log(addtional_html)
            //     tooltipText.innerHTML =tip.text.replace("This",addtional_html)+'<br>';
            //
            // }else {
            //     tooltipText.innerHTML = tip.text+'<br>';
            //
            // }
            tooltipText.innerHTML = tip.text+'<br>';
            tooltipButton.textContent = tip.button_name || "Got it";

            // Temporarily show the tooltip to get the correct height
            tooltip.style.display = 'block';
            tooltip.classList.add('visible');
            overlay.classList.add('visible');

            const rect = element.getBoundingClientRect();
            const viewportHeight = window.innerHeight;

            let tooltipTop, arrowDirection;
            // console.log(viewportHeight,rect.bottom ,tooltip.offsetHeight)
            if (rect.top + window.scrollY < viewportHeight / 2) {
                if ((viewportHeight-rect.bottom) < tooltip.offsetHeight + 20) {
                    // Element is in the upper half but too small to fit the tooltip below it
                    tooltipTop = rect.top + window.scrollY + (rect.height / 2) - (tooltip.offsetHeight / 2);
                    arrowDirection = 'top';
                } else {
                    // Element is in the upper half of the page
                    tooltipTop = rect.bottom + window.scrollY + 10; // Place below the element
                    arrowDirection = 'bottom';
                }
            } else {
                // Element is in the lower half of the page
                tooltipTop = rect.top + window.scrollY - tooltip.offsetHeight - 10; // Place above the element
                arrowDirection = 'top';
            }
            tooltip.style.top = `${tooltipTop}px`;
            tooltip.style.left = `${rect.left + window.scrollX + (rect.width / 2) - (tooltip.offsetWidth / 2)}px`;

            tooltip.classList.add(arrowDirection);



            // console.log('The z-index of the element is: ' + window.getComputedStyle(element).zIndex);
            // console.log('The z-index of the tool is: ' + window.getComputedStyle(tooltip).zIndex);
            // console.log('The z-index of the overlay is: ' + window.getComputedStyle(overlay).zIndex);
        }


        tooltipButton.addEventListener('click', async function () {

            const tip = tips[currentTipIndex];
            const element = document.evaluate(tip.xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
            if (element) {
                // Remove highlight class from element
                element.classList.remove('highlight');
            }
            tooltip.classList.remove('visible');

            overlay.classList.remove('visible');
            if (tip.command) {
                tip.command();
            }
            // Check the goal function
            if (tip.goal) {
                const goalAchieved = await tip.goal();
                if (!goalAchieved) return;
            }
            currentTipIndex++;
            if (currentTipIndex < tips.length) {
                setTimeout(() => {
                    tooltip.classList.remove('top', 'bottom', 'center');
                    showTip(currentTipIndex);
                }, 500); // Wait for the fade-out animation to complete
            } else {
                setTimeout(() => {
                    tooltip.style.display = 'none';
                    tooltip.classList.remove('top', 'bottom', 'center');
                }, 500); // Wait for the fade-out animation to complete
            }
        });

        // Show the first tip initially

            showTip(currentTipIndex);



}
function showCenteredTooltip(message) {
    return new Promise(resolve => {
        const overlay = document.getElementById('overlay');
        const tooltip = document.getElementById('centeredTooltip');
        const tooltipText = document.getElementById('tooltipText');
        const tooltipButton = document.getElementById('tooltipButton');

        tooltipText.innerHTML = message;

        tooltipButton.onclick = function() {
            tooltip.classList.remove('visible');
            overlay.classList.remove('visible');
            setTimeout(() => {
                tooltip.style.display = 'none';
                // overlay.style.display = 'none';
                resolve(true); // Resolve the promise when the button is clicked
            }, 500); // Wait for the fade-out animation to complete
        };

        // overlay.style.display = 'block';
        tooltip.style.display = 'block';
        setTimeout(() => {
            overlay.classList.add('visible');
            tooltip.classList.add('visible');
        }, 10); // Trigger the animation
    });
}
function get_innerhtml(xpath){
    // var xpath = "//div[@id='example']"; // 请根据实际情况修改XPath表达式

// 使用document.evaluate查找元素
    var result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);

// 获取找到的元素
    var element = result.singleNodeValue;

// 检查元素是否存在，并获取innerHTML
    if (element) {
        return element.innerHTML
    } else {
        console.log("Element not found");
    }
}
