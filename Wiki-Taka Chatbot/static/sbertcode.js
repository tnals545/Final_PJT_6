var pop;
var rs;
var anvoice;

window.onload = setup;

function setup() {
  pop = new Audio("/sounds/pop.mp3");
  pop.volume = 0.2;
  rs = new RiveScript();
  rs.loadFile(["/static/chatbotRiveScript.html"], on_load_success, on_load_error);
}

function on_load_success() {
  $("#message").removeAttr("disabled");
  $("#message").attr("placeholder", "Message");
  $("#message").focus();
  rs.sortReplies();
  getReply("start");
}

function on_load_error(err) {
  postReply(
    "Yikes, there was an error loading this bot. Refresh the page please."
  );
  console.log("Loading error: " + err);
}

function validate(event) {
   event.preventDefault();
};

function sendMessage() {
  var text = $("#message").val();
  if (text.length === 0) return validate(event);
  $("#message2").val(text);
  $("#message").val("");
  $("#message").attr("disabled",true);
  $("#dialogue").append(
    "<div class='user-row'><span class='user'>" +
      escapeHtml(text) +
      "</span></div>"
  );
  $("#dialogue").animate({ scrollTop: $("#dialogue")[0].scrollHeight }, 200);
  waitReply("pleasewait");
}

function getReply(text) {
  try {
    var reply = rs.reply("soandso", text);
    reply = reply.replace(/\n/g, "<br>");
    startpostReply(reply);
  } catch (e) {
    startpostReply(e.message + "\n" + e.line);
    console.log(e);
  }
}

function getReply2(text) {
  $("#message2").val("")
    var reply = $("#iframe1").contents().find('pre').html();
   if (reply == "Internal Server Error") {
    var reply = "현재 답변을 가져올 수 없습니다.";
    startpostReply(reply);
    } else {
    anvoice = new Audio("/sounds/answer.mp3")
    postReply(reply);
  }
  }

function selectoption(varRow) {
  document.forms[0].subtitle.options[varRow].selected = true ;
  $("#titlensub").submit();
  $("#subtitle").empty();
  $("#subbuttons").empty();
  $("#titlesubmit").remove();
}

function getsubtitle() {
  try {
    const json = $("#iframe2").contents().find('pre').html();
    const tst = JSON.parse(json);
    var title = tst.title;
    var pmes = tst.pmes
    if (title === '') {
      
    } else {
      $("#titlensub").append(
      "<input type='text' name='titlesubmit' id='titlesubmit' value='"
      + title +
      "' style='display:none'>"
    )
    var subtitle = tst.subtitle.split(", ");
    for (var i in subtitle) {
      $("#subtitle").append(
        "<option type='text' name='subtitle' value='"
        + subtitle[i] +
        "'>" + subtitle[i] + "</option>");
       $("#subbuttons").append(
        "<button name='subtitle' id='subtitle' onclick='selectoption("+ i +")'>"+ subtitle[i] +"</button>");    
      }
    }
    anvoice = new Audio("/sounds/answer.mp3")
    anvoice.volume = 0.4;
    postReply(pmes);
  } catch (e) {
    postReply(e.message + "\n" + e.line);
    console.log(e);
  }
  }

function waitReply(text) {
  try {
    var reply = rs.reply("soandso", text);
    reply = reply.replace(/\n/g, "<br>");
    postwaitReply(reply);
  } catch (e) {
    postwaitReply(e.message + "\n" + e.line);
    console.log(e);
  }
}

function postwaitReply(reply, delay) {
  if (!delay) delay = 800;
  var waitid = "disapear";
  setTimeout(function () {
    $("#dialogue").append(
      "<div class='bot-row' id='" +
        waitid +
        "'><img class= 'chatbotprofile' src = '/img/chatbotprofile.png' alt = 'noproflies'><span class='bot'>" +
        reply +
        "</span></div>"
    );
    pop.play();
    $("#" + waitid)
      .hide()
      .fadeIn(200);
    $("#dialogue").animate({ scrollTop: $("#dialogue")[0].scrollHeight }, 200);
  }, delay);
}

function startpostReply(reply, delay) {
  if (!delay) delay = 800;
  var rand = Math.round(Math.random() * 10000);;
  setTimeout(function () {
    $("#dialogue").append(
      "<div class='bot-row' id='" +
        rand +
        "'><img class= 'chatbotprofile' src = '/img/chatbotprofile.png' alt = 'noproflies'><span class='bot'>" +
        reply +
      "</span></div>"
    );
    pop.play();
    $("#" + rand)
      .hide()
      .fadeIn(200);
    $("#dialogue").animate({ scrollTop: $("#dialogue")[0].scrollHeight }, 200);
  }, delay);
}

function postReply(reply, delay) {
  if (!delay) delay = 800;
  var rand = Math.round(Math.random() * 10000);
  setTimeout(function () {
    $("#disapear").remove();
    $("#dialogue").append(
      "<div class='bot-row' id='" +
        rand +
        "'><img class= 'chatbotprofile' src = '/img/chatbotprofile.png' alt = 'noproflies'><span class='bot'>" +
        reply +
        "</span></div>"
    );
    anvoice.play();
    $("#message").attr("disabled", false);
    $("#" + rand)
      .hide()
      .fadeIn(200);
    $("#dialogue").animate({ scrollTop: $("#dialogue")[0].scrollHeight }, 200);
  }, delay);
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}