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
        Apply OffsetFromIndexes(Reference rf) {
            if (! IsNew(rf.VarRef.Domain)) {
                // 変数の定義域がない場合

                throw new Exception();
            }

            // 対象の変数参照をコピーする。
            rf = rf.Clone();

            // 変数の定義域をコピーする。
            Apply domain = (rf.VarRef.Domain as Apply).Clone();

            Term t = rf.Indexes[0];
            for(int dim = 1; dim < rf.Indexes.Length; dim++) {
                t = Add(Mul(domain.Args[dim],  t), rf.Indexes[dim]);
            }
            
            return t as Apply;
        }

        /*
            共通部分式除去

            スレッドとブロックのIDから添え字を計算。
            同じ添え字の計算式をまとめる。        
        */
        void CommonSubexpressionElimination(BlockStatement blc) {
            // ブロック文の中の多次元の添え字付きの変数参照のリスト
            List<Reference> refs = AllRefs(blc).Where(r => r.Indexes != null && r.Indexes.Length != 1).ToList();

            Dictionary<Apply, Variable> common_offsets = new Dictionary<Apply, Variable>(new TermEqualityComparer());

            foreach(Reference rf in refs) {
                Apply offset = OffsetFromIndexes(rf);

                Variable va;

                if(common_offsets.TryGetValue(offset, out va)) {

                }
                else {

                    va = new Variable(offset.ToString(), IntClass, null);
                    common_offsets.Add(offset, va);
                }
            }
        }

        /*
        
        */

        /*
        
        */

        /*
        
        */

        /*
         

        メモリの操作( CPU / GPU )
            確保
            解放
         
        カーネルの操作
            メモリーのコピー( CPU⇔GPU )
            起動
                引数のセット
                スレッドとブロックの割り当て
            同期
         
         
         */




    }
}
