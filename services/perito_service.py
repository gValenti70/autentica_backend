from core.db import db


def trova_periti_per_tipologia(tipologia):

    periti = list(
        db.aut_periti.find(
            {
                "tipologie": tipologia,
                "attivo": True
            }
        )
    )

    return periti
