import descritor_de_rede

dados_de_mobilidade = {
    0: "mobilidade/centroides_zonasSP.csv",
    1: "mobilidade/tab30_todos.csv",
    2: "mobilidade/tab24_coletivo.CSV",
    3: "mobilidade/tab25_individual.CSV",
    4: "mobilidade/tab26_motorizado.CSV",
    5: "mobilidade/tab27_ape.CSV",
    6: "mobilidade/tab28_bike.CSV",
    7: "mobilidade/tab29_naomotorizado.CSV"
}

descritor_de_rede.rmsp(dados_de_mobilidade)

