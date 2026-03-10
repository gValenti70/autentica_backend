import os


def genera_report_perizia(analisi, perito):

    template_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "templates",
        "perizia_mail.html"
    )

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    foto_html = ""

    for f in analisi.get("foto", []):
        foto_html += f'<img src="{f}" width="220" style="margin:10px;">'

    html = html.replace("{{ID_ANALISI}}", str(analisi.get("id_analisi")))
    html = html.replace("{{USER_ID}}", str(analisi.get("user_id")))
    html = html.replace("{{TIPOLOGIA}}", str(analisi.get("tipologia")))
    html = html.replace("{{PERCENTUALE}}", str(analisi.get("percentuale_contraffazione")))
    html = html.replace("{{FOTO}}", foto_html)

    return html
