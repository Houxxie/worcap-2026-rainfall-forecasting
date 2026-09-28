"""Testes isolados: dados sintéticos nunca entram no registro real."""
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
import copy
import json
import tempfile
import unittest
import numpy as np
import xarray as xr
from dados import LAT, LON, indices_psl
from registro import Registro, ContratoError, digest, deslocar
from avaliar import painel, avaliar

BASE = Path(__file__).resolve().parent
NOW = datetime(2026,9,28,20,0,tzinfo=timezone.utc)


class TestRegistro(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raiz=Path(self.temp.name)
        self.clock=patch('registro.agora',return_value=NOW)
        self.clock.start();self.addCleanup(self.clock.stop)
        self.clock2=patch('avaliar.agora',return_value=NOW)
        self.clock2.start();self.addCleanup(self.clock2.stop)
        self.r=Registro(self.raiz/'registro')
        self.plano=json.loads((BASE/'plano.json').read_text())
        self.protocolo=self.r.iniciar(self.plano)

    def pronto(self):
        arquivos={}
        for nome in self.plano['arquivos_modelo']:
            p=self.raiz/nome;p.write_bytes(b'fixture apenas sintetica')
            arquivos[nome]=p
        self.r.modelo(arquivos,dict(treino_inicio='1993-01',treino_fim='2022-12',protocolo_id=self.protocolo['id']))
        for n,regra in self.plano['fontes'].items():
            if regra['obrigatoria']:
                self.r.recibo(n,b'fixture sintetica '+n.encode(),
                    dict(validado=True,meses=[deslocar('2026-10',-regra['lag'])]),{'url':'fixture'})
        self.fp=self.raiz/'previsao.nc'
        a=np.ones((1,len(LAT),len(LON)),dtype=np.float32)
        ds=xr.Dataset({n:(('time','lat','lon'),a,{'units':'mm/day'}) for n in ['precipitacao','climatologia']},
            coords={'time':[np.datetime64('2026-10-01')],'lat':LAT,'lon':LON})
        ds.to_netcdf(self.fp)
        s=self.r.prontidao('2026-10')
        self.inf={k:s[k] for k in ['modelo','fontes','protocolo']}
        self.inf['previsao_sha256']=digest(self.fp.read_bytes())

    def test_coleta_e_integridade(self):
        e=self.r.recibo('nino34',b'abc',dict(validado=False,meses=[]),{'url':'fixture'})
        self.assertEqual(self.r.ler_objeto(e['dados']['objeto']),b'abc')
        p=self.r.pasta/'objetos'/(e['dados']['objeto']['sha256']+'.bin')
        p.write_bytes(b'abd')
        with self.assertRaises(ContratoError):self.r.eventos(conferir_objetos=True)

    def test_cadeia_detecta_evento_editado(self):
        self.r.adicionar('coleta_falhou',{'objetos':[]})
        p=sorted((self.r.pasta/'eventos').glob('*.json'))[-1]
        e=json.loads(p.read_text());e['tipo']='fonte_recebida'
        p.write_text(json.dumps(e))
        with self.assertRaises(ContratoError):self.r.eventos()

    def test_protocolo_imutavel(self):
        self.r.iniciar(self.plano)
        outro=copy.deepcopy(self.plano);outro['modelo']['arvores']=900
        with self.assertRaises(ContratoError):self.r.iniciar(outro)

    def test_faltantes_bloqueiam(self):
        s=self.r.prontidao('2026-10')
        self.assertFalse(s['pronto']);self.assertEqual(len(s['faltantes']),9)
        with self.assertRaises(ContratoError):self.r.congelar('2026-10','inexistente',{})

    def test_nao_aceita_mes_errado_ou_quarentena(self):
        self.r.recibo('nino34',b'abc',dict(validado=True,meses=['2026-08']),{'url':'fixture'})
        self.r.recibo('nino12',b'abc',dict(validado=False,meses=['2026-07']),{'url':'fixture'})
        self.assertNotIn('nino34',self.r.prontidao('2026-10')['fontes'])
        self.assertNotIn('nino12',self.r.prontidao('2026-10')['fontes'])

    def test_congelamento_e_reemissao_bloqueada(self):
        self.pronto()
        e=self.r.congelar('2026-10',self.fp,self.inf)
        self.assertEqual(e['registrado_em_utc'],NOW.isoformat())
        self.assertEqual(e['dados']['validacao']['pontos'],78561)
        with self.assertRaises(ContratoError):self.r.congelar('2026-10',self.fp,self.inf)

    def test_prazo_nao_pode_ser_retroagido(self):
        self.pronto()
        with patch('registro.agora',return_value=datetime(2026,10,1,tzinfo=timezone.utc)):
            with self.assertRaises(ContratoError):self.r.congelar('2026-10',self.fp,self.inf)
        self.assertFalse(any(e['tipo']=='previsao_congelada' for e in self.r.eventos()))

    def test_troca_fonte_apos_inferencia_bloqueia(self):
        self.pronto()
        self.r.recibo('tna',b'outra versao',dict(validado=True,meses=['2026-07']),{'url':'fixture'})
        with self.assertRaises(ContratoError):self.r.congelar('2026-10',self.fp,self.inf)

    def test_grid_e_valores_errados_bloqueiam(self):
        self.pronto()
        with xr.open_dataset(self.fp) as raw: ds=raw.load()
        ds['precipitacao'].values[0,0,0]=np.nan
        ds.to_netcdf(self.fp,mode='w')
        self.inf['previsao_sha256']=digest(self.fp.read_bytes())
        with self.assertRaises(ContratoError):self.r.congelar('2026-10',self.fp,self.inf)
        ds['precipitacao'].values[0,0,0]=1
        ds=ds.assign_coords(lon=LON+.01);ds.to_netcdf(self.fp,mode='w')
        self.inf['previsao_sha256']=digest(self.fp.read_bytes())
        with self.assertRaises(ContratoError):self.r.congelar('2026-10',self.fp,self.inf)

    def test_painel_inclui_meses_sem_emissao(self):
        d=painel(self.r)
        self.assertEqual(d['meses_planejados'],12)
        self.assertEqual(d['meses_emitidos'],0)
        self.assertEqual(d['metricas_primarias'],{})
        with patch('avaliar.agora',return_value=datetime(2026,11,2,tzinfo=timezone.utc)):
            self.assertEqual(painel(self.r)['meses'][0]['estado'],'nao_emitido_prazo_encerrado')

    def test_avaliacao_nao_libera_alvo_antes_da_hora(self):
        self.pronto();self.r.congelar('2026-10',self.fp,self.inf)
        with self.assertRaises(ContratoError):avaliar(self.r,'2026-10','inexistente',{})

    def test_avaliacao_primaria_nao_sobrescrita_por_revisao(self):
        self.pronto();self.r.congelar('2026-10',self.fp,self.inf)
        with xr.open_dataset(self.fp) as pred: truth=pred[['precipitacao']].rename({'precipitacao':'tp'}).load()
        obs=self.raiz/'obs.nc';truth.to_netcdf(obs)
        raw=truth.copy(deep=True);raw.tp.values[:]=.001
        raw.tp.attrs=dict(units='m',GRIB_stream='moda',GRIB_experimentVersionNumber='0001')
        bruto=self.raiz/'bruto.nc';raw.to_netcdf(bruto)
        info=dict(produto='ERA5 monthly_averaged_reanalysis',expver=1,mes_alvo='2026-10',
                  url='https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means',
                  arquivo_bruto=str(bruto))
        mature=datetime(2027,2,2,tzinfo=timezone.utc)
        with patch('registro.agora',return_value=mature),patch('avaliar.agora',return_value=mature):
            prim=avaliar(self.r,'2026-10',obs,info)
            self.assertTrue(prim['dados']['primaria'])
            truth.tp.values[:]=2;truth.to_netcdf(obs,mode='w')
            with self.assertRaises(ContratoError):avaliar(self.r,'2026-10',obs,info)
            raw.tp.values[:]=.002;raw.to_netcdf(bruto,mode='w')
            rev=avaliar(self.r,'2026-10',obs,info)
            self.assertFalse(rev['dados']['primaria'])
            self.assertEqual(painel(self.r)['metricas_primarias']['precipitacao/dominio']['rmse'],0)

    def test_derivado_exige_pais_da_mesma_fonte(self):
        e=self.r.recibo('nino34',b'abc',dict(validado=False,meses=[]),{'url':'fixture'})
        codigo=self.raiz/'adapter.py';codigo.write_text('fixture')
        with self.assertRaises(ContratoError):
            self.r.derivado('cfsv2',b'def',dict(validado=True,meses=['2026-09']),[e['id']],codigo)

    def test_parser_psl_nao_preenche_ausentes(self):
        b=('2026 2026\n2026 '+' '.join(['1']*7+['-99.99']*5)+'\n-99.99\nHadISST\n').encode()
        d=indices_psl(b)
        self.assertEqual(d['ultimo_mes'],'2026-07')
        self.assertEqual(d['ausentes'],5)
        with self.assertRaises(ContratoError):indices_psl(b'<html>login</html>')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TestRegistro)
    resultado=unittest.TextTestRunner(verbosity=2).run(suite)
    if not resultado.wasSuccessful():raise SystemExit(1)
