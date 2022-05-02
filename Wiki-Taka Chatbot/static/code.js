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

function sendMessage() {
  var text = $("#message").val();
  if (text.length === 0) return false;
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
  $("#message").val("")
  try {
    var reply = $("#iframe1").contents().find('pre').html();
    // reply = reply.replace(/\n/g, "<br>");
    if (reply === '{"detail":[{"loc":["body","message"],"msg":"field required","type":"value_error.missing"}]}') return false;
    anvoice = new Audio("/sounds/answer.mp3")
    postReply(reply);
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
  setTimeout(function() {
    $("#dialogue").append(
      "<div class='bot-row' id='" +
        waitid +
        "'><span class='bot'>" +
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

function startpostReply(reply, delay) {
  if (!delay) delay = 800;
  var rand = Math.round(Math.random() * 10000);;
  setTimeout(function () {
    $("#dialogue").append(
      "<div class='bot-row' id='" +
        rand +
        "'><span class='bot'>" +
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
        "'><span class='bot'>" +
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