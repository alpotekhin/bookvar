import {mkdtemp,readFile,writeFile,stat,rm,copyFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {describe,it,expect} from 'vitest';
import {materializeAsset} from '../adapter/materialize-asset.js';

describe('publication staging',()=>{
  it('copies by default and keeps the source independent',async()=>{
    const dir=await mkdtemp(join(tmpdir(),'bookvar-copy-test-'));
    try{
      const source=join(dir,'source'),dest=join(dir,'dest');
      await writeFile(source,'original');
      await materializeAsset(source,dest);
      expect((await stat(source)).ino).not.toBe((await stat(dest)).ino);
      await writeFile(dest,'changed');
      expect(await readFile(source,'utf8')).toBe('original');
    }finally{await rm(dir,{recursive:true});}
  });
  it('hardlinks only when requested, then permits an independent final copy',async()=>{
    const dir=await mkdtemp(join(tmpdir(),'bookvar-link-test-'));
    try{
      const source=join(dir,'source'),stage=join(dir,'stage'),output=join(dir,'output');
      await writeFile(source,'original');
      await materializeAsset(source,stage,true);
      expect((await stat(source)).ino).toBe((await stat(stage)).ino);
      expect(await readFile(stage,'utf8')).toBe('original');
      await copyFile(stage,output);
      expect((await stat(source)).ino).not.toBe((await stat(output)).ino);
      await writeFile(output,'changed');
      expect(await readFile(source,'utf8')).toBe('original');
    }finally{await rm(dir,{recursive:true});}
  });
  it('does not overwrite an existing destination in hardlink mode',async()=>{
    const dir=await mkdtemp(join(tmpdir(),'bookvar-link-existing-'));
    try{
      const source=join(dir,'source'),dest=join(dir,'dest');
      await writeFile(source,'source');await writeFile(dest,'kept');
      await expect(materializeAsset(source,dest,true)).rejects.toMatchObject({code:'EEXIST'});
      expect(await readFile(source,'utf8')).toBe('source');
      expect(await readFile(dest,'utf8')).toBe('kept');
    }finally{await rm(dir,{recursive:true});}
  });
});
