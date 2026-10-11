// Exact N3 CTC Viterbi implementation, .NET Framework C# x64 helper.
// Entire pipeline remains in _review. No BK lookup or timestamp mutation.
using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Globalization;

class N3Native {
    class Line {
        public int Cap, Count, C;
        public int[] Ids;
        public bool[] Skip;
        public int Syl;
    }
    class State {
        public float[][] Rows;
        public double Lead;
    }
    static float Neg = float.NegativeInfinity;
    static CultureInfo Inv = CultureInfo.InvariantCulture;
    static int[] ParseIds(string s) {
        string[] a=s.Split(',');
        int[] v=new int[a.Length];
        for(int j=0;j<a.Length;j++)v[j]=int.Parse(a[j],Inv);
        return v;
    }
    static float[][] Empty(Line[] lines) {
        float[][] rows=new float[lines.Length][];
        for(int i=0;i<lines.Length;i++){
            rows[i]=new float[lines[i].Count*lines[i].C];
            for(int j=0;j<rows[i].Length;j++)rows[i][j]=Neg;
        }
        return rows;
    }
    static State Clone(State p) {
        float[][] v=new float[p.Rows.Length][];
        for(int i=0;i<v.Length;i++)v[i]=(float[])p.Rows[i].Clone();
        return new State{Rows=v,Lead=p.Lead};
    }
    static void Step(float[] emission,int t,int vocab,Line[] lines,
                     State from,State to,byte[][] ptr,int offset) {
        int baseIndex=t*vocab;
        for(int line=0;line<lines.Length;line++){
            Line L=lines[line];int C=L.C;
            float[] prev=from.Rows[line];float[] dst=to.Rows[line];
            // All target states are only written if reached; overwrite each frame.
            for(int j=0;j<dst.Length;j++)dst[j]=Neg;
            byte[] parents=ptr==null?null:ptr[line];
            int ptrBase=offset*dst.Length;
            if(parents!=null)for(int j=0;j<dst.Length;j++)parents[ptrBase+j]=255;
            for(int state=0;state<L.Count;state++){
                int ix=state*C;
                float e=emission[baseIndex+L.Ids[state]];
                int previx=ix-C,skipix=ix-2*C;
                bool canSkip=state>=2 && L.Skip[state];
                for(int age=0;age<C;age++){
                    float val=prev[ix+age]; byte move=0;
                    if(state>0 && prev[previx+age]>val){
                        val=prev[previx+age];move=1;
                    }
                    if(canSkip && prev[skipix+age]>val){
                        val=prev[skipix+age];move=2;
                    }
                    if(float.IsNegativeInfinity(val))continue;
                    int nextAge=(age<L.Cap)?age+1:L.Cap;
                    int dstIdx=ix+nextAge;
                    // Use double to match numpy float32 addition followed by assignment.
                    float score=val+e;
                    if(score>dst[dstIdx]){
                        dst[dstIdx]=score;
                        if(parents!=null)parents[ptrBase+dstIdx]=(byte)(move+(age==L.Cap && L.Cap>0?8:0));
                    }
                }
            }
            double incoming;
            byte kind;
            if(line==0){
                incoming=from.Lead;kind=5;
            } else {
                Line P=lines[line-1];
                float[] old=from.Rows[line-1];
                float endBlank=old[(P.Count-1)*P.C+P.Cap];
                float endChar=L.Ids[0]!=P.Ids[P.Count-2]
                    ? old[(P.Count-2)*P.C+P.Cap]:Neg;
                if(endBlank>=endChar){incoming=endBlank;kind=3;}
                else {incoming=endChar;kind=4;}
            }
            float fromPrior=(float)(incoming+emission[baseIndex+L.Ids[0]]);
            if(fromPrior>dst[0]){
                dst[0]=fromPrior;
                if(parents!=null)parents[ptrBase]=kind;
            }
        }
        to.Lead=from.Lead+emission[baseIndex];
    }
    static Tuple<double,int[],int> Solve(float[] emissions,int T,int V,Line[] lines) {
        int block=180;
        State old=new State{Rows=Empty(lines),Lead=0.0};
        State newer=new State{Rows=Empty(lines),Lead=0.0};
        Dictionary<int,State> checkpoints=new Dictionary<int,State>();
        checkpoints[-1]=Clone(old);
        for(int t=0;t<T;t++){
            Step(emissions,t,V,lines,old,newer,null,0);
            State tmp=old;old=newer;newer=tmp;
            if(t%block==block-1)checkpoints[t]=Clone(old);
        }
        Line endL=lines[lines.Length-1];
        float finalBlank=old.Rows[lines.Length-1][(endL.Count-1)*endL.C+endL.Cap];
        float finalChar=old.Rows[lines.Length-1][(endL.Count-2)*endL.C+endL.Cap];
        float score=finalBlank>=finalChar?finalBlank:finalChar;
        if(float.IsInfinity(score) || float.IsNaN(score))
            throw new Exception("N3_NO_FINITE_PATH");
        int row=lines.Length-1;
        int state=finalBlank>=finalChar?endL.Count-1:endL.Count-2;
        int age=endL.Cap;
        int[] onsets=new int[lines.Length];
        for(int j=0;j<onsets.Length;j++)onsets[j]=-1;
        int target=T-1;
        while(target>=0){
            int previousEndpoint=(target/block)*block-1;
            State from=Clone(checkpoints[previousEndpoint]);
            State to=new State{Rows=Empty(lines),Lead=0.0};
            int length=target-previousEndpoint;
            byte[][] ptr=new byte[lines.Length][];
            for(int line=0;line<lines.Length;line++)
                ptr[line]=new byte[length*lines[line].Count*lines[line].C];
            for(int f=previousEndpoint+1;f<=target;f++){
                Step(emissions,f,V,lines,from,to,ptr,f-previousEndpoint-1);
                State tmp=from;from=to;to=tmp;
            }
            for(int frame=target;frame>previousEndpoint;frame--){
                Line L=lines[row];
                int pi=(frame-previousEndpoint-1)*L.Count*L.C+state*L.C+age;
                int code=ptr[row][pi];
                if(code==255)throw new Exception("N3_DEAD_TRACE at "+frame);
                if(code==3 || code==4 || code==5){
                    if(state!=0 || age!=0)throw new Exception("N3_BAD_ROW_ENTRY");
                    onsets[row]=frame;
                    if(code==5){
                        if(row!=0)throw new Exception("N3_BAD_PREFIX");
                        return Tuple.Create((double)score,onsets,lines.Length);
                    }
                    row-=1;
                    L=lines[row];age=L.Cap;
                    state=L.Count-(code==3?1:2);
                }else{
                    bool capped=code>=8;
                    int move=code%8;
                    if(move>2)throw new Exception("N3_INVALID_MOVE "+code);
                    state-=move;
                    age=capped?age:Math.Max(0,age-1);
                }
            }
            target=previousEndpoint;
        }
        throw new Exception("N3_TRACE_DID_NOT_REACH_START");
    }
    static float[] ReadNpy(string path,int T,int V){
        using(BinaryReader r=new BinaryReader(File.OpenRead(path))){
            byte[] sig=r.ReadBytes(6);
            if(sig.Length!=6 || sig[0]!=0x93 || Encoding.ASCII.GetString(sig,1,5)!="NUMPY")
                throw new Exception("EXPECTED_NUMPY");
            int major=r.ReadByte();r.ReadByte();
            int header=major==1?r.ReadUInt16():(int)r.ReadUInt32();
            string desc=Encoding.ASCII.GetString(r.ReadBytes(header));
            if(!desc.Contains("'<f4'") && !desc.Contains("'descr': '<f4'"))
                throw new Exception("EXPECTED_LE_FLOAT32 "+desc);
            if(!desc.Contains("False"))throw new Exception("EXPECTED_C_ARRAY");
            float[] x=new float[T*V];
            for(int j=0;j<x.Length;j++)x[j]=r.ReadSingle();
            if(r.BaseStream.Position!=r.BaseStream.Length)
                throw new Exception("NPY_SHAPE_SIZE_MISMATCH");
            return x;
        }
    }
    public static int Main(string[] args){
        try{
            if(args.Length!=2)throw new Exception("usage: n3_dp.exe input.txt output.txt");
            string[] input=File.ReadAllLines(args[0],Encoding.UTF8);
            string npy=input[0];
            string[] shape=input[1].Split(' ');
            int T=int.Parse(shape[0],Inv),V=int.Parse(shape[1],Inv);
            double spf=double.Parse(input[2],Inv);
            double threshold=double.Parse(input[3],Inv);
            int n=int.Parse(input[4],Inv);
            Line[] lines=new Line[n];
            for(int j=0;j<n;j++){
                string[] item=input[5+j].Split('|');
                int syl=int.Parse(item[0],Inv);
                int[] chars=ParseIds(item[1]);
                int cap=j<n-1?Math.Max(0,(int)Math.Ceiling(syl*threshold/spf-1e-10)-1):0;
                int[] labels=new int[chars.Length*2];
                bool[] skips=new bool[labels.Length];
                for(int k=0;k<chars.Length;k++){
                    labels[2*k]=chars[k];labels[2*k+1]=0;
                    if(k>0)skips[2*k]=chars[k]!=chars[k-1];
                }
                lines[j]=new Line{Cap=cap,C=cap+1,Count=labels.Length,Ids=labels,Skip=skips,Syl=syl};
            }
            float[] emissions=ReadNpy(npy,T,V);
            Tuple<double,int[],int> result=Solve(emissions,T,V,lines);
            string output=result.Item1.ToString("R",Inv)+"\n"
                +string.Join(",",result.Item2)+"\n";
            File.WriteAllText(args[1],output,Encoding.UTF8);
            return 0;
        }catch(Exception e){
            Console.Error.WriteLine(e.ToString());return 2;
        }
    }
}