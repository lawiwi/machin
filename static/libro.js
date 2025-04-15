const anteBtn = document.querySelector("#ante-Btn");
const sigBtn = document.querySelector("#sig-Btn");
const libro = document.querySelector("#libro");

const pagina1 = document.querySelector("#p1");
const pagina2 = document.querySelector("#p2");
const pagina3 = document.querySelector("#p3");


anteBtn.addEventListener("click", goPrevPage);
sigBtn.addEventListener("click", goNextPage);


let currentLocation = 1;
let numpaginas = 3;
let maxLocation = numpaginas + 1;

function abrirlibro() {
    libro.style.transform = "translateX(50%)";
    anteBtn.style.transform = "translateX(-180px)";
    sigBtn.style.transform = "translateX(180px)";
}

function cerrarlibro(isAtBeginning) {
    if(isAtBeginning) {
        libro.style.transform = "translateX(0%)";
    } else {
        libro.style.transform = "translateX(100%)";
    }
    
    anteBtn.style.transform = "translateX(0px)";
    sigBtn.style.transform = "translateX(0px)";
}

function goNextPage() {
    if(currentLocation < maxLocation) {
        switch(currentLocation) {
            case 1:
                abrirlibro();
                pagina1.classList.add("flipped");
                pagina1.style.zIndex = 1;
                break;
            case 2:
                pagina2.classList.add("flipped");
                pagina2.style.zIndex = 2;
                break;
            case 3:
                pagina3.classList.add("flipped");
                pagina3.style.zIndex = 3;
                cerrarlibro(false);
                break;
            default:
                throw new Error("unkown state");
        }
        currentLocation++;
    }
}

function goPrevPage() {
    if(currentLocation > 1) {
        switch(currentLocation) {
            case 2:
                cerrarlibro(true);
                pagina1.classList.remove("flipped");
                pagina1.style.zIndex = 3;
                break;
            case 3:
                pagina2.classList.remove("flipped");
                pagina2.style.zIndex = 2;
                break;
            case 4:
                abrirlibro();
                pagina3.classList.remove("flipped");
                pagina3.style.zIndex = 1;
                break;
            default:
                throw new Error("unkown state");
        }

        currentLocation--;
    }
}