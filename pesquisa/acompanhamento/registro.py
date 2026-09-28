"""Registro local auditável. Não é assinatura nem carimbo de tempo independente."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re


class ContratoError(ValueError):
    pass


def exigir(ok, mensagem):
    if not ok:
        raise ContratoError(mensagem)


def agora():
    return datetime.now(timezone.utc)


def instante(valor):
    t = datetime.fromisoformat(str(valor).replace('Z', '+00:00'))
    exigir(t.tzinfo is not None, 'Horário sem fuso.')
    return t.astimezone(timezone.utc)


def mes(valor):
    exigir(bool(re.fullmatch(r'\d{4}-\d{2}', str(valor))), 'Mês deve ser YYYY-MM.')
    t = datetime.strptime(valor, '%Y-%m').replace(tzinfo=timezone.utc)
    return t


def deslocar(valor, n):
    t = mes(valor)
    k = t.year * 12 + t.month - 1 + n
    return f'{k // 12:04d}-{k % 12 + 1:02d}'


def canonico(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def escrever_novo(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Nunca sobrescrever silenciosamente um artefato registrado.
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


class Registro:
    def __init__(self, pasta):
        self.pasta = Path(pasta).resolve()
        self.pasta.mkdir(parents=True, exist_ok=True)
        for nome in ('objetos', 'eventos'):
            (self.pasta / nome).mkdir(exist_ok=True)

    @contextmanager
    def transacao(self):
        lock = self.pasta / '.escrita.lock'
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise ContratoError('Registro em uso ou lock de execução interrompida. Não remover automaticamente.') from None
        os.close(fd)
        try:
            yield
        finally:
            lock.unlink()

    def guardar(self, dados):
        h = digest(dados)
        p = self.pasta / 'objetos' / (h + '.bin')
        if p.exists():
            exigir(p.read_bytes() == dados, 'Objeto alterado no registro.')
        else:
            try:
                escrever_novo(p, dados)
            except FileExistsError:
                exigir(p.read_bytes() == dados, 'Colisão de objeto.')
        return dict(sha256=h, bytes=len(dados))

    def ler_objeto(self, ref):
        h = ref['sha256']
        exigir(bool(re.fullmatch('[0-9a-f]{64}', h)), 'Hash inválido.')
        p = self.pasta / 'objetos' / (h + '.bin')
        exigir(p.is_file(), 'Objeto ausente: ' + h)
        b = p.read_bytes()
        exigir(len(b) == ref['bytes'] and digest(b) == h, 'Objeto modificado: ' + h)
        return b

    def eventos(self, conferir_objetos=False):
        out, anterior, ultimo = [], None, None
        for i, p in enumerate(sorted((self.pasta / 'eventos').glob('*.json')), 1):
            e = json.loads(p.read_text(encoding='utf-8'))
            h = e.pop('id')
            exigir(digest(canonico(e)) == h, 'Evento alterado: ' + p.name)
            exigir(p.name == f'{i:08d}_{h}.json', 'Sequência de eventos incompleta.')
            exigir(e['sequencia'] == i and e['anterior'] == anterior, 'Cadeia de eventos inválida.')
            t = instante(e['registrado_em_utc'])
            exigir(ultimo is None or t >= ultimo, 'Relógio retrocedeu; emissão bloqueada.')
            e['id'] = h
            if conferir_objetos:
                for ref in e['dados'].get('objetos', []):
                    self.ler_objeto(ref)
            out.append(e)
            anterior, ultimo = h, t
        return out

    def _adicionar(self, tipo, dados, prazo_exclusivo=None):
        # Chamador mantém transacao; não aceita data de emissão fornecida pelo usuário.
        existentes = self.eventos()
        t = agora()
        exigir(not existentes or t >= instante(existentes[-1]['registrado_em_utc']), 'Relógio retrocedeu.')
        exigir(prazo_exclusivo is None or t < prazo_exclusivo, 'Prazo ultrapassado antes de registrar a emissão.')
        e = dict(sequencia=len(existentes) + 1,
                 anterior=existentes[-1]['id'] if existentes else None,
                 registrado_em_utc=t.isoformat(), tipo=tipo, dados=dados)
        e['id'] = digest(canonico(e))
        escrever_novo(self.pasta/'eventos'/f"{e['sequencia']:08d}_{e['id']}.json", canonico(e))
        return e

    def adicionar(self, tipo, dados):
        with self.transacao():
            return self._adicionar(tipo, dados)

    def obter(self, ident, tipo=None):
        encontrados = [e for e in self.eventos() if e['id'] == ident]
        exigir(len(encontrados) == 1, 'Evento desconhecido.')
        e = encontrados[0]
        exigir(tipo is None or e['tipo'] == tipo, 'Tipo de evento incorreto.')
        return e

    def iniciar(self, plano):
        with self.transacao():
            anteriores = [e for e in self.eventos() if e['tipo'] == 'protocolo']
            if anteriores:
                exigir(len(anteriores) == 1 and anteriores[0]['dados']['plano'] == plano,
                       'Plano mudou. Abrir outro registro/experimento.')
                return anteriores[0]
            exigir(agora() < mes(plano['primeiro_mes']), 'Protocolo não pode iniciar com mês passado/em curso.')
            return self._adicionar('protocolo', dict(plano=plano, objetos=[]))

    def protocolo(self):
        e = [e for e in self.eventos() if e['tipo'] == 'protocolo']
        exigir(len(e) == 1, 'Inicialize o protocolo primeiro.')
        return e[0]

    def recibo(self, fonte, dados, metadados, transporte):
        """Horário é o da gravação real, nunca o mtime do arquivo ou nominal do mês."""
        exigir(fonte in self.protocolo()['dados']['plano']['fontes'], 'Fonte fora do protocolo.')
        with self.transacao():
            ref = self.guardar(dados)
            return self._adicionar('fonte_recebida', dict(fonte=fonte, objeto=ref, objetos=[ref],
                inspecao=metadados, transporte=transporte, primeira_publicacao_utc=None,
                evidencia='bytes recebidos nesta coleta; não comprova primeira publicação histórica'))

    def modelo(self, arquivos, auditoria):
        """Registra o pacote final; não treina e não atesta sozinho o algoritmo."""
        p = self.protocolo()
        plano = p['dados']['plano']
        exigir(set(arquivos) == set(plano['arquivos_modelo']), 'Pacote de modelo incompleto ou inesperado.')
        exigir(auditoria['treino_inicio'] == plano['treino_inicio'] and
               auditoria['treino_fim'] == plano['treino_fim'], 'Treinamento fora do plano.')
        exigir(auditoria.get('protocolo_id') == p['id'], 'Auditoria não corresponde ao protocolo.')
        with self.transacao():
            exigir(not any(e['tipo'] == 'modelo_congelado' for e in self.eventos()), 'Modelo já congelado neste acompanhamento.')
            refs = {nome: self.guardar(Path(path).read_bytes()) for nome, path in arquivos.items()}
            return self._adicionar('modelo_congelado', dict(protocolo=p['id'], arquivos=refs,
                objetos=list(refs.values()), auditoria=auditoria))

    def derivado(self, fonte, dados, metadados, pais, codigo):
        exigir(bool(pais) and len(pais)==len(set(pais)), 'Originais ausentes/duplicados.')
        exigir(fonte in self.protocolo()['dados']['plano']['fontes'], 'Fonte fora do protocolo.')
        with self.transacao():
            for ident in pais:
                e=self.obter(ident,'fonte_recebida')
                exigir(e['dados']['fonte']==fonte, 'Original pertence a outra fonte.')
                self.ler_objeto(e['dados']['objeto'])
            ref=self.guardar(dados)
            cod=self.guardar(Path(codigo).read_bytes())
            return self._adicionar('fonte_recebida',dict(fonte=fonte,objeto=ref,objetos=[ref,cod],
                inspecao=metadados,pais=pais,codigo=cod,primeira_publicacao_utc=None,
                transporte=dict(metodo='transformacao local dos originais registrados'),
                evidencia='aquisição dos originais e processamento com código preservado'))

    def prontidao(self, alvo):
        mes(alvo)
        protocolo = self.protocolo()
        p = protocolo['dados']['plano']
        exigir(p['primeiro_mes'] <= alvo <= p['ultimo_mes'], 'Mês fora do acompanhamento.')
        ev = self.eventos(conferir_objetos=True)
        fontes, faltantes = {}, []
        for fonte, regra in p['fontes'].items():
            if not regra['obrigatoria']:
                continue
            esperado = deslocar(alvo, -regra['lag'])
            candidatos = [e for e in ev if e['tipo'] == 'fonte_recebida'
                and e['dados']['fonte'] == fonte
                and esperado in e['dados']['inspecao'].get('meses', [])
                and e['dados']['inspecao'].get('validado') is True
                and instante(e['registrado_em_utc']) < mes(alvo)]
            if candidatos:
                # Política fixa: última versão recebida e validada antes da emissão.
                fontes[fonte] = candidatos[-1]['id']
            else:
                faltantes.append(f'{fonte}: mês {esperado}')
        modelos = [e for e in ev if e['tipo'] == 'modelo_congelado']
        if not modelos:
            faltantes.append('modelo: pacote final do controle com defasagens ainda não registrado')
        if agora() >= mes(alvo):
            faltantes.append('prazo: mês já começou; não retroagir emissão')
        return dict(mes_alvo=alvo, pronto=not faltantes, faltantes=faltantes, fontes=fontes,
                    modelo=modelos[0]['id'] if modelos else None,
                    protocolo=protocolo['id'], previsao_emitida=False)

    def congelar(self, alvo, arquivo, inferencia):
        """Aceita previsão da inferência validada; nenhuma opção de backdate/overwrite."""
        from dados import validar_previsao
        with self.transacao():
            exigir(not any(e['tipo'] == 'previsao_congelada' and e['dados']['mes_alvo'] == alvo
                           for e in self.eventos()), 'Previsão já existe; revisão exige outra série explicitamente separada.')
            status = self.prontidao(alvo)
            exigir(status['pronto'], 'Emissão bloqueada: ' + '; '.join(status['faltantes']))
            exigir(inferencia.get('modelo') == status['modelo'] and
                   inferencia.get('fontes') == status['fontes'] and
                   inferencia.get('protocolo') == status['protocolo'], 'Inferência não usa as versões selecionadas.')
            b = Path(arquivo).read_bytes()
            exigir(inferencia.get('previsao_sha256') == digest(b), 'Hash da inferência diverge do arquivo.')
            resumo = validar_previsao(arquivo, alvo)
            ref = self.guardar(b)
            # Nova verificação do relógio depois da validação/gravação dos bytes.
            exigir(agora() < mes(alvo), 'Prazo ultrapassado durante a preparação.')
            return self._adicionar('previsao_congelada', dict(mes_alvo=alvo, protocolo=status['protocolo'],
                modelo=status['modelo'], fontes=status['fontes'], inferencia=inferencia,
                previsao=ref, objetos=[ref], validacao=resumo, prazo_exclusivo_utc=mes(alvo).isoformat(),
                carimbo_independente=False), prazo_exclusivo=mes(alvo))

    def exportar_resumo(self, destino):
        ev = self.eventos(conferir_objetos=True)
        d = dict(eventos=len(ev), ponta=ev[-1]['id'] if ev else None,
                 conferido_em_utc=agora().isoformat(),
                 garantia='cadeia e hashes locais; não assinatura nem armazenamento inviolável',
                 fontes=[dict(id=e['id'], recebido_em_utc=e['registrado_em_utc'], **e['dados'])
                         for e in ev if e['tipo'] == 'fonte_recebida'],
                 previsoes=[e for e in ev if e['tipo'] == 'previsao_congelada'])
        Path(destino).write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding='utf-8')
        return d
