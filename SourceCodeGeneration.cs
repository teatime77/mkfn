using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace MkFn {
    public partial class MkFn {

        /*
            添え字から配列の先頭からのオフセットの計算式を得る。
        */
        public static Term OffsetFromIndexes(Reference rf) {
            if (!IsNew(rf.VarRef.Domain)) {
                // 変数の定義域がない場合

                throw new Exception();
            }

            // 対象の変数参照をコピーする。
            rf = rf.Clone();

            // 変数の定義域をコピーする。
            Apply domain = (rf.VarRef.Domain as Apply).Clone();

            Term t = rf.Indexes[0];
            for (int dim = 1; dim < rf.Indexes.Length; dim++) {
                t = Add(Mul(domain.Args[dim], t), rf.Indexes[dim]);
            }

            return t;
        }

        /*
CUDAとCの違い
・ 出力先 \\src\\CUDA
・ ファイル拡張子 .cu / .cpp
・ #include
・ cudaMalloc、cudaFree
・ cudaStreamCreate、cudaStreamDestroy
・ カーネル関数の本体
   ・ ヘッダー行 __global__  inline
   ・ 代入先の添え字
     ・ threadIdxとblockIdxから計算
     ・ カーネル関数の引数に含める。
・ カーネル関数の起動
・ 
・ 
・ 
        
        */

        /*
        
        */

        /*
        
        */

        /*
         
        バッチ
            配列のアクセス
            パラメータの更新
                定義域が同じパラメータをまとめる。
                    カーネル関数の生成
                カーネル関数の起動の生成

        検証
            CalculatedField
                単一
                複数
            ParameterField
                単一
                複数
            レイヤー
                単一
                複数
         */
    }
}
