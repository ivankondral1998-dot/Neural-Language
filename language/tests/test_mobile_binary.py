"""NLMOBIN1 roundtrip, independent oracle, corruption and no-overwrite tests."""
from __future__ import annotations

import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mobile_pack import PackingError, convert
from tests.test_mobile_infer import write_model


class MobileBinaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='nl_mobile_binary_')
        cls.folder=Path(cls.temp.name)
        cls.executables={}
        for name in ('g++','clang++'):
            if shutil.which(name):
                output=cls.folder/('gcc' if name=='g++' else 'clang')
                build=subprocess.run([name,'-std=c++17','-O2','-Wall','-Wextra','-Werror',
                                      str(ROOT/'mobile_infer.cpp'),'-o',str(output)],
                                     capture_output=True,text=True,timeout=90)
                if build.returncode:raise RuntimeError(build.stderr)
                cls.executables[name]=output
        if not cls.executables:raise unittest.SkipTest('no C++17 toolchain')

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'temp'):cls.temp.cleanup()

    def infer(self,binary,model,inp,extra=()):
        p=subprocess.run([str(binary),'--model',str(model),'--input',','.join(map(repr,inp)),*extra],
                         capture_output=True,text=True,timeout=12)
        if p.returncode:return None,p
        return [float(v) for v in p.stdout.split('values=')[1].splitlines()[0].split(',')],p

    def test_100_roundtrips_vs_text_and_python(self):
        rng=random.Random(20261108)
        for i in range(100):
            n=rng.randint(1,6)
            d=n
            layers=[]
            for _ in range(rng.randint(1,5)):
                out=rng.randint(1,9)
                quant=rng.choice((0,1))
                ws=[rng.randint(-127,127) if quant else rng.uniform(-1,1) for j in range(out*d)]
                layers.append((out,rng.choice((0,1,2)),quant,
                              [rng.uniform(-0.5,0.5) for _ in range(out)],
                              ws,[rng.choice((0,1)) for _ in ws]))
                d=out
            text=self.folder/f'random_{i}.nlw'
            packed=self.folder/f'random_{i}.nlmb'
            reference=write_model(text,n,layers)
            meta=convert(text,packed)
            self.assertEqual(meta['weights'],sum(len(y[4]) for y in layers))
            data=[rng.uniform(-2,2) for _ in range(n)]
            oracle=reference(data)
            for cc,binary in self.executables.items():
                with self.subTest(i=i,cc=cc):
                    old,pr=self.infer(binary,text,data)
                    new,pr2=self.infer(binary,packed,data)
                    self.assertIsNotNone(new,pr2.stderr)
                    self.assertEqual(old,new)
                    for a,b in zip(new,oracle):self.assertAlmostEqual(a,b,delta=1e-12*(1+abs(b)))

    def test_reject_corruption_header_crc_and_trailing_bytes(self):
        source=self.folder/'tamper_source.nlw'
        out=self.folder/'tamper_original.nlmb'
        write_model(source,2,[(2,1,1,[0.1,-0.2],[30,0,2,-10],[1,0,1,1])])
        convert(source,out)
        pristine=out.read_bytes()
        binary=next(iter(self.executables.values()))
        mutations=[
            pristine[:8]+b'\x02'+pristine[9:],         # wrong version
            pristine[:24]+b'\xff'*8+pristine[32:],    # bad payload size
            pristine[:32]+b'\x00'*4+pristine[36:],    # wrong CRC
            pristine[:40]+bytes([pristine[40]^1])+pristine[41:], # modified payload
            pristine[:-1], pristine+b'X',
            pristine[:16]+b'\x09'+pristine[17:],      # layers out of bounds
            pristine[:36]+b'\x01'+pristine[37:],      # reserved field
        ]
        for i,blob in enumerate(mutations):
            p=self.folder/f'tampered_{i}.nlmb';p.write_bytes(blob)
            got,run=self.infer(binary,p,[0.2,0.3])
            with self.subTest(i=i):
                self.assertIsNone(got)
                self.assertNotEqual(run.returncode,0)
                self.assertIn('NEURAL MOBILE ERROR',run.stderr)

    def test_reject_invalid_int8_code_and_mask_padding_with_valid_crc(self):
        import zlib
        text=self.folder/'edge.nlw';mobile=self.folder/'edge.nlmb'
        write_model(text,1,[(1,0,1,[0],[0],[1])])
        convert(text,mobile)
        original=bytearray(mobile.read_bytes())
        binary=next(iter(self.executables.values()))
        # layout: header 40, layer 16, bias 8, code 1, gate 1
        for where,value in [(64,128),(65,255)]:
            corrupt=bytearray(original)
            corrupt[where]=value
            corrupt[32:36]=struct.pack('<I',zlib.crc32(corrupt[40:]))
            p=self.folder/f'bad_code_{where}.nlmb';p.write_bytes(corrupt)
            prediction,proc=self.infer(binary,p,[0.2])
            self.assertIsNone(prediction,proc.stdout)

    def test_compact_size_for_medium_int8_network(self):
        rng=random.Random(32)
        inputs=128;out=256
        n=inputs*out
        text=self.folder/'medium.nlw';binary=self.folder/'medium.nlmb'
        reference=write_model(text,inputs,[(out,1,1,[0]*out,
                                          [rng.randint(-127,127) for _ in range(n)],
                                          [1 if i%13 else 0 for i in range(n)])])
        meta=convert(text,binary)
        self.assertLess(meta['mobile_bytes'] / meta['source_bytes'],0.30)
        old,p=self.infer(next(iter(self.executables.values())),text,[0.2]*inputs)
        new,p=self.infer(next(iter(self.executables.values())),binary,[0.2]*inputs)
        self.assertIsNotNone(new,p.stderr)
        self.assertEqual(old,new)
        self.assertEqual(len(new),out)
        for a,b in zip(new,reference([0.2]*inputs)):
            self.assertAlmostEqual(a,b,delta=1e-12)
        print('MEDIUM_INT8_SIZE',meta['source_bytes'],meta['mobile_bytes'],
              'reduction_pct',round(100*(1-meta['mobile_bytes']/meta['source_bytes']),2))

    def test_no_overwrite_or_repack_corrupt_source(self):
        path=self.folder/'preservation.nlw';out=self.folder/'preservation.nlmb'
        write_model(path,1,[(1,0,0,[0],[0.1],[1])])
        first=convert(path,out)
        frozen=out.read_bytes()
        with self.assertRaises(PackingError):convert(path,out)
        self.assertEqual(frozen,out.read_bytes())
        with self.assertRaises(PackingError):convert(path,path)
        self.assertEqual(first['source_bytes'],path.stat().st_size)
        path.write_text(path.read_text().replace('NLMODEL2','NLMODEL3'))
        with self.assertRaises(PackingError):convert(path,self.folder/'never.nlmb')
        self.assertFalse((self.folder/'never.nlmb').exists())


if __name__=='__main__':unittest.main(verbosity=2)
