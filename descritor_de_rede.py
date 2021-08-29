import math

import numpy as np
from random import randint
from random import sample
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import igraph as ig
from matplotlib import pylab
import pandas as pd
from scipy.stats import ks_2samp as kolmogorov_smirnov_similarity


def numero_de_vertices(grafo):
    return grafo.vcount()


def numero_de_arestas(grafo):
    return grafo.ecount()


def lista_de_arestas(grafo):
    return grafo.get_edgelist()


def fluxo_total(grafo):
    return sum(grafo.es['weight'])


def componentes(grafo):
    return grafo.components()


def obter_grau_do_vertice(grafo, index):
    return grafo.vs[index].degree()


def obter_distribuicao_dos_graus(grafo):
    dg = []
    for i in range(0, 342):
        dg.append(grafo.vs[i].degree())
    return dg


def obter_nome_do_componente(grafo, id_componente):
    try:
        nome_do_componente = grafo.vs[componentes(grafo)[id_componente]]['label']
    except IndexError:
        print(f'ID de componente fora do intervalo 0 - {len(componentes(grafo))}')
        print("Tente outro id")
        return None
    return nome_do_componente


def numero_de_componentes(grafo):
    return len(grafo.components())


def tamanho_do_componente_gigante(grafo):
    return float(max(grafo.components().sizes()))


def deletar_vertice(grafo, indice_do_vertice):
    return grafo.delete_vertices(indice_do_vertice)


# Retorna um indice gerados aleatoriamente e sem repetição dentro do intervalo de 0 a n.
def gerador_de_falhas_aleatorias(n):
    return sample(range(0, n), 1)


def obter_nome_do_vertice(grafo, indice):
    return grafo.vs[indice]['label']


# @param numero_total_de_vertices: numero total de vertices do grafo original
# P(f) é o tamanho do componente gigante apos remover uma taxa f de vértices.
def p(grafo, numero_total_de_vertices):
    qtde_de_vertices_removidos = numero_total_de_vertices - numero_de_vertices(grafo)
    taxa_percentual_de_remocao = float(qtde_de_vertices_removidos / numero_total_de_vertices)  # f em P(f)
    try:
        tcg = tamanho_do_componente_gigante(grafo)  # se rede está vazia igraph lança exceção ValueError.
    except ValueError:
        tcg = 0
    return taxa_percentual_de_remocao, tcg


def aplicar_falhas_aleatorias(grafo):
    n = numero_de_vertices(grafo)  # ex.: n = 342
    historico_de_falhas_aleatorias = [(p(grafo, n), fluxo_total(grafo))]  # ex.: retorna: [((0, p(0)), fluxo_remanesc)]
    while numero_de_vertices(grafo) > 0:
        falha_aleatoria = gerador_de_falhas_aleatorias(numero_de_vertices(grafo))
        # print(f'Ponto de ataque: {obter_nome_do_vertice(grafo, falha_aleatoria)}')
        deletar_vertice(grafo, falha_aleatoria)
        historico_de_falhas_aleatorias.append((p(grafo, n), fluxo_total(grafo)))
    return historico_de_falhas_aleatorias  # retorna: [((f, p(f)), w)]


# dado uma rede, atacamos os nós de maior grau até que não sobre nenhum. Retorna-se a taxa de nós removidos e o tcg.
def aplicar_ataques_coordenados_grau(grafo):
    n = numero_de_vertices(grafo)
    historico_de_ataque = [(p(grafo, n), fluxo_total(grafo))]
    while numero_de_vertices(grafo) > 0:
        ataque_maior_grau = encontrar_maior_grau(grafo)
        deletar_vertice(grafo, ataque_maior_grau)
        historico_de_ataque.append((p(grafo, n), fluxo_total(grafo)))
    return historico_de_ataque


def aplicar_ataques_coordenados_forca(grafo):
    n = numero_de_vertices(grafo)
    historico_de_ataque = [(p(grafo, n), fluxo_total(grafo))]
    while numero_de_vertices(grafo) > 0:
        ataque_maior_fluxo = encontrar_maior_fluxo(grafo)
        deletar_vertice(grafo, ataque_maior_fluxo)
        historico_de_ataque.append((p(grafo, n), fluxo_total(grafo)))
    return historico_de_ataque


def aplicar_ataques_coordenados_betweenness(grafo):
    n = numero_de_vertices(grafo)
    historico_de_ataque = [(p(grafo, n), fluxo_total(grafo))]
    while numero_de_vertices(grafo) > 0:
        ataque_maior_betweenness = encontrar_maior_betweenness(grafo)
        deletar_vertice(grafo, ataque_maior_betweenness)
        historico_de_ataque.append((p(grafo, n), fluxo_total(grafo)))
    return historico_de_ataque


# retorna o vértice de maior grau na rede.
def encontrar_maior_grau(grafo):
    lista_de_graus = grafo.degree()
    maior_grau = max(lista_de_graus)
    return lista_de_graus.index(maior_grau)


# retorna o vértice de maior betweenness na rede.
def encontrar_maior_betweenness(grafo):
    lista_de_betweenness = grafo.betweenness()
    maior_betweenness = max(lista_de_betweenness)
    return lista_de_betweenness.index(maior_betweenness)


# retorna o vértice de maior fluxo na rede.
def encontrar_maior_fluxo(grafo):
    n = numero_de_vertices(grafo)
    fluxo = []
    fluxo_vertice = 0
    # para cada vértice da rede, verifique o peso de cada incidência
    for indice_vertice in range(n):
        for indice_incidencia in range(len(grafo.vs[indice_vertice].incident())):
            fluxo_vertice += grafo.vs[indice_vertice].incident()[indice_incidencia]['weight']
        fluxo.append(fluxo_vertice)
        fluxo_vertice = 0
    maior_fluxo = max(fluxo)
    return fluxo.index(maior_fluxo)


def obter_fluxo_total_de_cada_vertice(grafo):
    n = numero_de_vertices(grafo)
    fluxo = []
    fluxo_vertice = 0
    # para cada vértice da rede, verifique o peso de cada incidência
    for indice_vertice in range(n):
        for indice_incidencia in range(len(grafo.vs[indice_vertice].incident())):
            fluxo_vertice += grafo.vs[indice_vertice].incident()[indice_incidencia]['weight']
        fluxo.append(fluxo_vertice)
        fluxo_vertice = 0
    return fluxo


# análise da taxa de remoção e do tamanho do componente gigante
def obter_grafico_de_robustez(historico):  # histórico contém [ ((f, tcg), w)]
    eixo_x = []
    eixo_y = []

    # pegando o valor de P(0) da formula P(f)/P(0) a partir do meu historico de perturbação da rede
    ((_, p_infinito_zero), w) = historico[0]

    # gerando uma lista com os valores de x (representados por f) e y (representados pelo resultado de P(f)/P(0))
    for ((f, pf), _) in historico:
        eixo_x.append(f)
        eixo_y.append(pf / float(p_infinito_zero))

    # plotagem do gráfico
    fig, ax = plt.subplots()

    ax.set_xlabel(r'$f$', fontsize=14)
    ax.set_ylabel(r'$P_\infty(f) / P_\infty(0)$', fontsize=14)
    ax.text(-0.1, 1.1, "Robustez RMSP", transform=ax.transAxes, size=20, weight='bold')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.plot(eixo_x, eixo_y, 'c')
    print("Preparando o gráfico")
    plt.show()
    return None


# auxilia em simulações
def cria_copias_da_rede(grafo, nro_de_copias):
    copias_da_rede = []
    for id_copia in range(nro_de_copias):
        copias_da_rede.append(grafo.copy())
    return copias_da_rede


def simular_falhas_aleatorias(grafo, qtde_de_simulacoes):
    resultados_das_simulacoes = []
    copias_da_rede = cria_copias_da_rede(grafo, qtde_de_simulacoes)
    # print(f'Quantidade de copias da rede: {len(copias_da_rede)}')
    # print(copias_da_rede)
    for rede in copias_da_rede:
        resultados_das_simulacoes.append(aplicar_falhas_aleatorias(rede))
    return resultados_das_simulacoes


def simular_ataques_coordenados_grau(grafo, qtde_de_simulacoes):
    resultados_das_simulacoes = []
    copias_da_rede = cria_copias_da_rede(grafo, qtde_de_simulacoes)
    for rede in copias_da_rede:
        resultados_das_simulacoes.append(aplicar_ataques_coordenados_grau(rede))
    return resultados_das_simulacoes


def simular_ataques_coordenados_forca(grafo, qtde_de_simulacoes):
    resultados_das_simulacoes = []
    copias_da_rede = cria_copias_da_rede(grafo, qtde_de_simulacoes)
    for rede in copias_da_rede:
        resultados_das_simulacoes.append(aplicar_ataques_coordenados_forca(rede))
    return resultados_das_simulacoes


def simular_ataques_coordenados_betweenness(grafo, qtde_de_simulacoes):
    resultados_das_simulacoes = []
    copias_da_rede = cria_copias_da_rede(grafo, qtde_de_simulacoes)
    for rede in copias_da_rede:
        resultados_das_simulacoes.append(aplicar_ataques_coordenados_betweenness(rede))
    return resultados_das_simulacoes


def obter_historico_medio(resultados_das_simulacoes):
    historico_medio = []
    soma_f = 0
    soma_pf = 0
    soma_w = 0
    qtde_de_simulacoes = len(resultados_das_simulacoes)
    iteracao = 0
    # print(f'Quantidade de simulacoes: {qtde_de_simulacoes}')
    while iteracao < 343:
        for simulacao in range(qtde_de_simulacoes):
            ((f, pf), w) = resultados_das_simulacoes[simulacao][iteracao]
            soma_f += f
            soma_pf += pf
            soma_w += w
            # print(f'Simulacao: {simulacao} - Iteracao: {iteracao} -> soma_f: {soma_f}, soma_pf: {soma_pf}')
        historico_medio.append(
            ((soma_f / qtde_de_simulacoes, soma_pf / qtde_de_simulacoes), soma_w / qtde_de_simulacoes))
        soma_f = 0
        soma_pf = 0
        soma_w = 0
        iteracao += 1
    return historico_medio


# Métricas de Rede
def densidade(grafo):
    return grafo.density()


def grau_medio(grafo):
    return np.mean(grafo.degree())


def forca_media(grafo):
    return np.mean(grafo.strength(weights="weight"))


def diametro(grafo):
    return np.mean(grafo.get_diameter())


def caminho_minimo_medio_da_rede(grafo):
    # o caminho mínimo médio da rede é a média da média dos caminhos mínimos existentes para cada vértice.
    soma_dos_caminhos_minimos_do_vertice = np.zeros(grafo.vcount())
    # comprimento_do_caminho_minimo_para_vertice = np.zeros(grafo.vcount())
    vertices_desconexos_do_vertice = []  # vértices desconexos ao vertice X da rede.
    index = 0
    for caminhos in (grafo.shortest_paths()):
        #  comprimento_do_caminho_minimo_para_vertice[index] = len(caminhos)
        vertice_desconexo = 0
        for caminho in caminhos:
            if caminho != np.inf:  # vértices alcansáveis.
                soma_dos_caminhos_minimos_do_vertice[index] += caminho
                vertice_desconexo += 1
            else:
                vertices_desconexos_do_vertice.append((index, vertice_desconexo))
                vertice_desconexo += 1
                # Quais print(vertices_desconexos_do_vertice[index])
                # Quantos print(len(vertices_desconexos_do_vertice[index]))
        index += 1

    media_dos_caminhos_minimos_do_vertice = []
    # print(f'Soma dos caminhos minimos: {soma_dos_caminhos_minimos_do_vertice}')
    for i in range(grafo.vcount()):
        media_dos_caminhos_minimos_do_vertice.append(soma_dos_caminhos_minimos_do_vertice[i] / grafo.vcount())
    # np.seterr('raise')
    return np.mean(media_dos_caminhos_minimos_do_vertice)


def coeficiente_de_clusterizacao(grafo):
    return grafo.transitivity_avglocal_undirected()


def assortatividade(grafo):
    return grafo.assortativity_degree()


def ponto_de_articulacao(grafo):
    return grafo.articulation_points()


def arvore_geradora(grafo):
    return grafo.spanning_tree()


def plotar_grafo(grafo):
    lyt = []
    for i in range(grafo.vcount()):
        lyt.append((grafo.vs[i]["X"], grafo.vs[i]["Y"] * (-1)))

    style = {"vertex_size": grafo.vs["size"], "edge_width": 0.4, "layout": lyt, "bbox": (1200, 1400), "margin": 30}

    ig.plot(grafo, **style)


def gerar_distribuicao_dos_graus(grafo):
    dist_graus = grafo.degree()
    return np.histogram(dist_graus, density=True)


def gerar_grafico(titulo, etiqueta_x, etiqueta_y, dados_x, dados_y):
    fig = plt.figure()
    plt.title(titulo)
    plt.plot(dados_y[:-1], dados_x)
    plt.xlabel(etiqueta_x, fontsize=12)
    plt.ylabel(etiqueta_y, fontsize=12)
    # fig.savefig('rmsp_' + titulo + '.jpg')
    plt.show()
    return 1


def gerar_grafico_baselog(titulo, etiqueta_x, etiqueta_y, dados_x, dados_y):
    fig = plt.figure()
    plt.title(titulo)
    dados_x = np.log10(dados_x)
    dados_y = np.log10(dados_y)
    print(f'x: {dados_x}')
    print(f'y: {dados_y}')
    plt.plot(dados_y[:-1], dados_x)
    plt.xlabel(etiqueta_x, fontsize=12)
    plt.ylabel(etiqueta_y, fontsize=12)
    # fig.savefig('rmsp_' + titulo + '.jpg')
    plt.show()
    return 1


def gerar_distribuicao_de_poisson():
    poisson = np.random.poisson(lam=15, size=342)
    h = np.histogram(poisson, density=True)
    prob = h[0]
    val = h[1][:-1]
    return val, prob, poisson


def fit_poisson(g, distribuicao_graus, modal):
    val, prob, poisson = gerar_distribuicao_de_poisson()
    pk, k = distribuicao_graus

    k = k[:-1]  # removendo o "ponto extra" que o np.histogram coloca ao final.

    # compatibilizando os pontos de poisson para a mesma escala da distribuição em análise
    min_val_wanted = min(k)
    min_val_existing = min(val)
    max_val_wanted = max(k)
    max_val_existing = max(val)
    min_prob_wanted = min(pk)
    min_prob_existing = min(prob)
    max_prob_wanted = max(pk)
    max_prob_existing = max(prob)

    (new_val, new_prob) = [
        min_val_wanted + \
        ((val - min_val_existing) *
         (max_val_wanted - min_val_wanted) /
         (max_val_existing - min_val_existing)),

        min_prob_wanted + \
        ((prob - min_prob_existing) *
         (max_prob_wanted - min_prob_wanted) /
         (max_prob_existing - min_prob_existing))
    ]

    # mudando para base log10 os valores de poisson escalonados
    log_val = np.log10(new_val)
    log_prob = np.log10(new_prob)

    # mudando para base log10 os valores da distribuição em análise
    log_k = np.log10(k)
    log_pk = np.log10(pk)

    # calculando a similaridade entre as duas distribuições.
    # estatistica, valor = kolmogorov_smirnov_similarity(poisson, g.degree())
    # "\nÍndice de Kolmogorov-Smirnov: \n(est=" + str("{:.4f}".format(estatistica)) + "valor=" +
    # str("{:.4f}".format(valor)) + ")", transform = ax.transAxes, size = 20, weight = 'bold'

    # plotando os valores
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$k$', fontsize=14)
    ax.set_ylabel(r'$P(k)$', fontsize=14)
    ax.text(-0.1, 1.1, "Poisson Fitness - modal: " + modal, transform=ax.transAxes, size=20, weight='bold')

    # índice de similaridade em relacão ao eixo x
    x_statistics, x_pvalue = kolmogorov_smirnov_similarity(log_k, log_val)
    # índice de similaridade em relacão ao eixo y
    y_statistics, y_pvalue = kolmogorov_smirnov_similarity(log_pk, log_prob)

    # goodness_x = "x = (" + str("{:.2f}".format(x_statistics)) + ", " + str("{:.2f}".format(x_pvalue)) + ")"
    goodness_pk = "(" + str("{:.2f}".format(y_statistics)) + ", " + str("{:.2f}".format(y_pvalue)) + ")"

    plt.plot(log_val, log_prob, label="Distribuição de Poisson", color="green")
    plt.plot(log_k, log_pk, label="RMSP " + goodness_pk, color="blue")

    plt.legend()  # para ativar a legenda.
    plt.margins(x=0.02, y=0.02)
    fig.savefig('distribuicoes/PoissonFitness_kolmogorov_test' + modal + '.jpg')
    plt.show()
    return 1


# retorna um gráfico sobre o tamanho do componente gigante ao longo das perturbações aplicadas à rede.
# @param uma lista com o resultado dos ataques
def analisar_tamanho_do_componente_gigante(historico_de_perturbacoes, versao):
    falhas_aleatorias = historico_de_perturbacoes[0]
    ataques_coordenados_grau = historico_de_perturbacoes[1]
    ataques_coordenados_forca = historico_de_perturbacoes[2]
    ataques_coordenados_betweenness = historico_de_perturbacoes[3]

    # x = f -> taxa de remoção dos nós
    # y = P(f)/P(0) -> tamanho do componente gigante

    # falhas aleatórias
    eixo_x1 = []
    eixo_y1 = []
    # ataques coordenados por grau
    eixo_x2 = []
    eixo_y2 = []
    # ataques coordenados por forca
    eixo_x3 = []
    eixo_y3 = []
    # ataques coordenados por betweeness
    eixo_x4 = []
    eixo_y4 = []

    # pegando o valor de P(0) da formula P(f)/P(0) a partir do meu historico de perturbação da rede
    ((_, p_infinito_zero), _) = falhas_aleatorias[0]
    """ Montando os eixos das abcissas (f) e das ordenadas (P(f)/P(0)) para as falhas aleatórias """
    for ((f, pf), _) in falhas_aleatorias:
        eixo_x1.append(f)
        eixo_y1.append(pf / float(p_infinito_zero))

    robustez_falhas_aleatorias = sum(eixo_y1[1:]) / (len(eixo_y1) - 1)
    print(f'Falhas Aleatórias - sum(eixo_y1[1:]: {sum(eixo_y1[1:])}')
    print(f'Falhas Aleatórias - len(eixo_y1): {len(eixo_y1[1:])}')

    """ Montando os eixos das abcissas (f) e das ordenadas (P(f)/P(0)) para as ataques coordenados por grau"""
    for ((f, pf), _) in ataques_coordenados_grau:
        eixo_x2.append(f)
        eixo_y2.append(pf / float(p_infinito_zero))

    robustez_ataques_coordenados_grau = sum(eixo_y2[1:]) / (len(eixo_y2) - 1)

    """ Montando os eixos das abcissas (f) e das ordenadas (P(f)/P(0)) para as ataques coordenados por força"""
    for ((f, pf), _) in ataques_coordenados_forca:
        eixo_x3.append(f)
        eixo_y3.append(pf / float(p_infinito_zero))

    robustez_ataques_coordenados_forca = sum(eixo_y3[1:]) / (len(eixo_y3) - 1)

    """ Montando os eixos das abcissas (f) e das ordenadas (P(f)/P(0)) para as ataques coordenados por betweenness"""
    for ((f, pf), _) in ataques_coordenados_betweenness:
        eixo_x4.append(f)
        eixo_y4.append(pf / float(p_infinito_zero))

    robustez_ataques_coordenados_betweeness = sum(eixo_y4[1:]) / (len(eixo_y4) - 1)

    # plotagem do gráfico
    fig, ax = plt.subplots()

    ax.set_xlabel(r'$f$', fontsize=14)
    ax.set_ylabel(r'$P_\infty(f) / P_\infty(0)$', fontsize=14)
    ax.text(-0.1, 1.1, "Robustez RMSP " + versao, transform=ax.transAxes, size=20, weight='bold')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(eixo_y1)])

    plt.plot(eixo_x1, eixo_y1, label="Falhas Aleatórias (" + str("{:.4f}".format(robustez_falhas_aleatorias)) + ")")
    plt.plot(eixo_x2, eixo_y2, label="Ataque por grau (" + str("{:.4f}".format(robustez_ataques_coordenados_grau)) +
                                     ")")
    plt.plot(eixo_x3, eixo_y3, label="Ataque por força (" + str("{:.4f}".format(robustez_ataques_coordenados_forca)) +
                                     ")")
    plt.plot(eixo_x4, eixo_y4, label="Ataque por betweenness (" + str("{:.4f}".format(
        robustez_ataques_coordenados_betweeness)) + ")")

    plt.legend()
    plt.margins(x=0.02, y=0.02)
    plt.savefig("rmsp_tcg" + versao + ".png")
    plt.show()

    return None


# retorna um gráfico sobre o tamanho do componente gigante ao longo das perturbações aplicadas à rede.
def analisar_fluxo_total_remanescente(historico_de_perturbacoes, versao):
    falhas_aleatorias = historico_de_perturbacoes[0]
    ataques_coordenados_grau = historico_de_perturbacoes[1]
    ataques_coordenados_forca = historico_de_perturbacoes[2]
    ataques_coordenados_betweenness = historico_de_perturbacoes[3]

    # x = f -> taxa de remoção dos nós
    # y = W -> fluxo total remanescente

    # falhas aleatórias
    eixo_x1 = []
    eixo_y1 = []
    # ataques coordenados por grau
    eixo_x2 = []
    eixo_y2 = []
    # ataques coordenados por forca
    eixo_x3 = []
    eixo_y3 = []
    # ataques coordenados por betweeness
    eixo_x4 = []
    eixo_y4 = []

    # pegando o valor de w a partir do meu historico de perturbação da rede
    ((_, _), w) = falhas_aleatorias[0]
    """ Montando os eixos das abcissas (f) e das ordenadas (W) para as falhas aleatórias """
    for ((f, _), w) in falhas_aleatorias:
        eixo_x1.append(f)
        eixo_y1.append(w)

    robustez_falhas_aleatorias = sum(eixo_y1[1:]) / (len(eixo_y1) - 1)

    """ Montando os eixos das abcissas (f) e das ordenadas (W) para as ataques coordenados por grau"""
    for ((f, _), w) in ataques_coordenados_grau:
        eixo_x2.append(f)
        eixo_y2.append(w)

    robustez_ataques_coordenados_grau = sum(eixo_y2[1:]) / (len(eixo_y2) - 1)

    """ Montando os eixos das abcissas (f) e das ordenadas (W) para as ataques coordenados por força"""
    for ((f, _), w) in ataques_coordenados_forca:
        eixo_x3.append(f)
        eixo_y3.append(w)

    robustez_ataques_coordenados_forca = sum(eixo_y3[1:]) / (len(eixo_y3) - 1)

    """ Montando os eixos das abcissas (f) e das ordenadas (W) para as ataques coordenados por betweenness"""
    for ((f, _), w) in ataques_coordenados_betweenness:
        eixo_x4.append(f)
        eixo_y4.append(w)

    robustez_ataques_coordenados_betweeness = sum(eixo_y4[1:]) / (len(eixo_y4) - 1)

    # plotagem do gráfico
    fig, ax = plt.subplots()

    ax.set_xlabel(r'$f$', fontsize=14)
    ax.set_ylabel(r'$||W||$', fontsize=14)
    ax.text(-0.1, 1.1, "Robustez RMSP " + versao, transform=ax.transAxes, size=20, weight='bold')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(eixo_y1)])

    plt.plot(eixo_x1, eixo_y1, label="Falhas Aleatórias (" + str("{:.4f}".format(robustez_falhas_aleatorias)) + ")")
    plt.plot(eixo_x2, eixo_y2, label="Ataque por grau (" + str("{:.4f}".format(robustez_ataques_coordenados_grau)) +
                                     ")")
    plt.plot(eixo_x3, eixo_y3, label="Ataque por força (" + str("{:.4f}".format(robustez_ataques_coordenados_forca)) +
                                     ")")
    plt.plot(eixo_x4, eixo_y4, label="Ataque por betweenness (" + str("{:.4f}".format(
        robustez_ataques_coordenados_betweeness)) + ")")

    plt.legend()
    plt.margins(x=0.02, y=0.02)
    plt.savefig("rmsp_w" + versao + ".png")
    plt.show()

    return None


# betweeness nas arestas.
# bridges

###################################################################
# Versão 2
def ataques_coordenados(grafo, modo_de_ataque):
    n = numero_de_vertices(grafo)
    historico_de_ataque = [(p(grafo, n), fluxo_total(grafo))]
    cod_id = 0
    while numero_de_vertices(grafo) > 0:
        ataque_maior_metrica = grafo.vs.find(codigo=modo_de_ataque[cod_id][0]).index
        deletar_vertice(grafo, ataque_maior_metrica)
        historico_de_ataque.append((p(grafo, n), fluxo_total(grafo)))
        cod_id += 1
    return historico_de_ataque


# não analisa o estado atual da rede para efetuar a estratégia de ataque.
# ataques feitos com base no estado inicial da rede.
def analise_de_robustez_versao2(g, modal):
    N = g.vcount()
    codigo = np.linspace(0, N - 1, N)
    g.vs['codigo'] = codigo

    degree = g.degree()
    metrica = degree  # pode ser qualquer outra

    dicionario_da_rede = []
    for i in range(len(metrica)):
        dicionario_da_rede.append((codigo[i], float(metrica[i])))

    # stat_array = [ (0,2.0), (1,1.0), (2,1.0) ]

    esquema = [('codigo', int), ('metrica', float)]

    dicionario_da_rede = np.array(dicionario_da_rede, dtype=esquema)
    dicionario_da_rede = np.sort(dicionario_da_rede, order='metrica')
    grau = np.flip(dicionario_da_rede)  # mapa de ataque por grau
    ###########################################################################
    fluxo = obter_fluxo_total_de_cada_vertice(g)
    metrica = fluxo
    dic_de_fluxo = []
    for i in range(len(metrica)):
        dic_de_fluxo.append((codigo[i], float(metrica[i])))

    # stat_array = [ (0,2.0), (1,1.0), (2,1.0) ]

    dic_de_fluxo = np.array(dic_de_fluxo, dtype=esquema)
    dic_de_fluxo = np.sort(dic_de_fluxo, order='metrica')
    fluxo = np.flip(dic_de_fluxo)  # mapa de ataque por fluxo

    #############################################################################
    betweenness = g.betweenness()
    metrica = betweenness
    dic_de_betweenness = []
    for i in range(len(metrica)):
        dic_de_betweenness.append((codigo[i], float(metrica[i])))

    # stat_array = [ (0,2.0), (1,1.0), (2,1.0) ]

    dic_de_betweenness = np.array(dic_de_betweenness, dtype=esquema)
    dic_de_betweenness = np.sort(dic_de_betweenness, order='metrica')
    betweenness = np.flip(dic_de_betweenness)  # mapa de ataque por betweenness

    historico0 = aplicar_falhas_aleatorias(g.copy())
    historico1 = ataques_coordenados(g.copy(), modo_de_ataque=grau)
    historico2 = ataques_coordenados(g.copy(), modo_de_ataque=fluxo)
    historico3 = ataques_coordenados(g.copy(), modo_de_ataque=betweenness)

    historico_de_perturbacoes = [
        historico0,
        historico1,
        historico2,
        historico3
    ]

    # analisar_tamanho_do_componente_gigante(historico_de_perturbacoes, versao="_" + modal + "_v2")
    analisar_fluxo_total_remanescente(historico_de_perturbacoes, versao="_" + modal + "_v2.1")


# analisa o estado atual da rede para efetuar a estratégia de ataque.
def analise_de_robustez_versao1(g, modal):
    falhas_aleatorias = simular_falhas_aleatorias(g, 1)
    ataques_coordenados_grau = simular_ataques_coordenados_grau(g, 1)
    ataques_coordenados_forca = simular_ataques_coordenados_forca(g, 1)
    ataques_coordenados_betweenness = simular_ataques_coordenados_betweenness(g, 1)

    historico_medio_falhas_aleatorias = obter_historico_medio(falhas_aleatorias)
    historico_medio_ataques_grau = obter_historico_medio(ataques_coordenados_grau)
    historico_medio_ataques_forca = obter_historico_medio(ataques_coordenados_forca)
    historico_medio_ataques_betweenness = obter_historico_medio(ataques_coordenados_betweenness)

    historico_de_perturbacoes = [
        historico_medio_falhas_aleatorias,
        historico_medio_ataques_grau,
        historico_medio_ataques_forca,
        historico_medio_ataques_betweenness
    ]

    # analisar_tamanho_do_componente_gigante(historico_de_perturbacoes, versao="_" + modal + "_v1")
    analisar_fluxo_total_remanescente(historico_de_perturbacoes, versao="_" + modal + "_v1.1")


def rmsp(arquivo):
    path = 1
    while path < 8:
        dado_de_mobilidade = pd.read_csv(arquivo.get(path), delimiter=';', header=None)
        bairros_de_sp = pd.read_csv(arquivo.get(0), delimiter=';', encoding="latin1")
        deslocamentos = dado_de_mobilidade.values

        modal = arquivo.get(path).split('/')[1].split('.')[0].split('_')[1]

        g = ig.Graph.Weighted_Adjacency(deslocamentos.tolist(), attr="weight", mode=ig.ADJ_MAX)
        g.to_undirected()
        g.es['weight'] = deslocamentos[deslocamentos.nonzero()]
        g.vs['label'] = bairros_de_sp['NomeZona'].tolist()
        # coordendas
        g.vs["X"] = bairros_de_sp['COORD_X'].tolist()
        g.vs["Y"] = bairros_de_sp['COORD_Y'].tolist()

        # normalizacao dos graus dos vértices
        # deixar o tamanho do nó de acordo com o grau
        node_size = g.degree()
        menor_grau_existente = min(node_size)
        maior_grau_existente = max(node_size)
        # min and max desejáveis
        max_size_wanted = 100
        min_size_wanted = 5
        # normalização do tamanho do nó
        node_size = np.array(node_size)
        node_size = min_size_wanted + \
                    ((node_size - menor_grau_existente) *
                     (max_size_wanted - min_size_wanted) /
                     (maior_grau_existente - menor_grau_existente))
        # saving new sizes
        g.vs["size"] = node_size.tolist()

        # plotar_grafo(g)
        # g.write_graphml("rmsp_modal_" + modal + ".GraphML")

        # distribuicao_dos_graus = gerar_distribuicao_dos_graus(g)
        # # avalia graficamente a correspondência da distr. de todos os modais com a distr de poisson.
        # fit_poisson(g, distribuicao_dos_graus, modal)

        analise_de_robustez_versao1(g, modal=modal)
        analise_de_robustez_versao2(g, modal=modal)
        path += 1
