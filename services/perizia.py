def genera_report_perizia(analisi, perito):

    foto_html = ""

    for f in analisi.get("foto", []):
        foto_html += f'<img src="{f}" width="220" style="margin:10px;">'

    html = f"""
<html>
<body style="font-family:Arial;background:#f4f6f8;padding:30px">

<div style="
background:white;
padding:30px;
border-radius:10px;
max-width:800px;
box-shadow:0 2px 8px rgba(0,0,0,0.1)
">

<h2>Richiesta perizia - Autentica</h2>

<p>
È stata effettuata una nuova analisi che richiede verifica peritale.
</p>

<hr>

<h3>Dati analisi</h3>

<b>ID analisi:</b> {analisi.get("id_analisi")}<br>
<b>Utente:</b> {analisi.get("user_id")}<br>
<b>Tipologia:</b> {analisi.get("tipologia")}<br>

<br>

<h3>Risultato AI</h3>

<b>Probabilità contraffazione:</b> {analisi.get("percentuale_contraffazione")}%


<br><br>

<h3>Immagini analisi</h3>

{foto_html}

<br><br>

<a href="https://autentica.app/analisi/{analisi.get("id_analisi")}"
style="
background:#2c7be5;
color:white;
padding:12px 20px;
text-decoration:none;
border-radius:6px;
font-weight:bold
">

Apri analisi completa

</a>

</div>

</body>
</html>
"""

    return html
