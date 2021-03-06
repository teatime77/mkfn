﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MkFn {
    public partial class MkFn {

        /*
            LINQの微分
        */
        Term DifferentialLINQ(LINQ lnq, Reference r1, Dictionary<Variable, Variable> var_tbl) {
            Debug.Assert(lnq.Aggregate != null);

            Dictionary<Reference, Dictionary<Reference, Term>> rs = new Dictionary<Reference, Dictionary<Reference, Term>>();
            bool exact = false;

            // r1と同じ変数を参照する変数参照のリスト
            var refs  = All<Reference>(lnq.Select).Where(x => x.VarRef == r1.VarRef);

            foreach(Reference r2 in refs) {
                if (rs.Keys.Any(r => r.Eq(r2))) {
                    // 処理済みの場合

                    continue;
                }

                if (r1.Eq(r2)) {
                    // 一致する場合

                    exact = true;
                }
                else {
                    // 一致しない添え字がある場合

                    Dictionary<Reference, Term> pairs = new Dictionary<Reference, Term>();
                    for (int i = 0; i < r1.Indexes.Length; i++) {
                        if (!r1.Indexes[i].Eq(r2.Indexes[i])) {
                            // 添え字が一致しない場合

                            if (!(r2.Indexes[i] is Reference)) {
                                // 代入候補の変数参照の添え字が変数参照でない場合

                                throw new Exception();
                            }
                            else {
                                // 両方の添え字が変数参照の場合

                                Reference r3 = r2.Indexes[i] as Reference;
                                var linq_eq_vars = lnq.Variables.Where(va => va == r3.VarRef);
                                if (linq_eq_vars.Any()) {
                                    // LINQの変数の場合

                                    Variable va = linq_eq_vars.First();
                                    Debug.Assert(! pairs.Keys.Any(r => r.VarRef == va));
                                    pairs.Add(new Reference(va), r1.Indexes[i]);
                                }
                                else {
                                    // LINQの変数でない場合

                                    throw new Exception();
                                }
                            }
                        }
                    }

                    rs.Add(r2, pairs);
                }
            }

            Debug.Assert(!(exact && rs.Any()), "完全一致の変数参照と代入で一致の変数参照の両方がある場合は未対応");
            if (!(exact || rs.Any())) {
                // 完全一致や代入で一致の変数参照がない場合

                return Zero();
            }

            // LINQをコピーします。
            LINQ lnq1 = lnq.Clone(var_tbl);

            Term lnq_select;

            if (exact) {
                // 完全一致の変数参照がある場合

                // select句を微分します。
                lnq_select = lnq1.Select;
            }
            else {
                // 代入で一致の変数参照がある場合

                Debug.Assert(rs.Keys.Count == 1, "代入で一致の変数参照は1種類のみ実装");
                Dictionary<Reference, Term> subst_tbl = rs.First().Value;
                Debug.Assert(subst_tbl.Count == lnq.Variables.Length, "LINQの全変数に代入します。");

                // LINQのselect句の変数参照に代入します。
                lnq_select = Subst(lnq1.Select, subst_tbl, var_tbl);
            }

            // LINQのselect句を微分します。
            Term dif1 = Differential(lnq_select, r1, var_tbl);

            if (lnq.Aggregate.Name == "Sum") {
                // 集計関数が総和の場合

                return dif1;
            }
            else {
                // 集計関数が総和でない場合

                // 総和以外のLINQの微分は未実装
                throw new Exception();
            }
        }

        /*
            微分
        */
        Term Differential(Term t1, Reference r1, Dictionary<Variable, Variable> var_tbl_up) {
            if (t1 is Reference) {
                // 変数参照の場合

                if (t1.Eq(r1)) {
                    return One();
                }
                else {
                    return Zero();
                }
            }
            else if (t1 is Number) {
                // 数値定数の場合

                return Zero();
            }

            Dictionary<Variable, Variable> var_tbl = (var_tbl_up == null ? new Dictionary<Variable, Variable>() : new Dictionary<Variable, Variable>(var_tbl_up));
            if (t1 is Apply) {
                // 関数適用の場合

                Apply app = t1 as Apply;

                Term[] diffs = (from t in app.Args select Differential(t, r1, var_tbl)).ToArray();

                if (app.FunctionApp.VarRef == AddFnc) {
                    // 加算の場合

                    return Add(diffs);
                }
                else if (app.FunctionApp.VarRef == MulFnc) {
                    // 乗算の場合

                    Term[] args = new Term[app.Args.Length];
                    foreach (int i in Range(app.Args.Length)) {
                        args[i] = Mul((from j in Range(app.Args.Length) select (i == j ? diffs[i] : app.Args[j].Clone(var_tbl))).ToArray());
                    }

                    return Add(args);
                }
                else if (app.FunctionApp.VarRef == MaxPoolFnc) {
                    return new Apply(MaxPoolPrimeFnc, app.Args.Select(x => x.Clone(var_tbl)).ToArray());
                }
                else if (app.FunctionApp.Name == "σ") {

                    Term[] args = (from t in app.Args select t.Clone(var_tbl)).ToArray();
                    return Mul(new Apply(new Reference(σ_prime), args), diffs[0]);
                }
                else if (app.FunctionApp.Name == "tanh") {

                    Term[] args = (from t in app.Args select t.Clone(var_tbl)).ToArray();
                    return Mul(new Apply(new Reference(tanh_prime), args), diffs[0]);
                }
                else if (app.FunctionApp.Name == "Mat" || app.FunctionApp.Name == "Row" || app.FunctionApp.Name == "C") {
                    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 未実装 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    return Zero();
                }
                else {
                    Debug.Assert(false);
                }
            }
            else if (t1 is LINQ) {
                // LINQの場合

                return DifferentialLINQ(t1 as LINQ, r1, var_tbl);
            }
            else {
                Debug.Assert(false);
            }

            Debug.Assert(false);
            return null;
        }

        /*
            変数に項を代入します。
        */
        Term Subst(Term t1, Dictionary<Reference, Term> subst_tbl, Dictionary<Variable, Variable> var_tbl = null) {
            return TraverseRep(t1,
                delegate (object obj, out object ret) {
                    ret = obj;
                    if (obj is Reference) {
                        // 変数参照の場合

                        var vref = from r in subst_tbl.Keys where r.Eq(obj) select r;
                        if (vref.Any()) {

                            ret = subst_tbl[vref.First()].Clone(var_tbl);
                            return true;
                        }
                    }

                    return false;
                }) as Term;
        }

        /*
            数式の簡約化
        */
        Term SimplifyExpression(Term t1) {
            return TraverseRep(t1,
                delegate (object obj, out object ret) {
                    ret = obj;

                    if (obj is Apply) {
                        // 関数適用の場合

                        Apply app = obj as Apply;

                        // 引数を簡約化します。
                        Term[] args1 = (from t in app.Args select SimplifyExpression(t)).ToArray();


                        if (app.IsAdd() || app.IsMul()) {

                            List<Term> args2 = new List<Term>();

                            foreach (Term t2 in args1) {
                                if (t2 is Apply && (t2 as Apply).FunctionApp.VarRef == app.FunctionApp.VarRef) {
                                    // 引数が同じ演算の場合

                                    args2.AddRange(t2.AsApply().Args);
                                }
                                else {
                                    // 引数が加算や減算でない場合

                                    args2.Add(t2);
                                }
                            }

                            for (int i = 0; i < args2.Count; i++) {

                                for (int j = i + 1; j < args2.Count;) {
                                    if (args2[i].EqBody(args2[j])) {
                                        // 本体が同じ項がある場合

                                        if (app.IsAdd()) {
                                            // 加算の場合

                                            // 係数を加算します。
                                            args2[i].Value += args2[j].Value;
                                        }
                                        else {
                                            // 乗算の場合

                                            // 係数を乗算します。
                                            args2[i].Value *= args2[j].Value;
                                        }

                                        // 同じ項を取り除きます。
                                        args2.RemoveAt(j);
                                    }
                                    else {
                                        // 同じ項がない場合

                                        j++;
                                    }
                                }
                            }

                            if (app.IsAdd()) {
                                // 加算の場合

                                // 係数が0の項を除きます。
                                Term[] args3 = (from t in args2 where t.Value != 0 select t).ToArray();

                                switch (args3.Length) {
                                case 0:
                                    ret = Zero();
                                    return true;

                                case 1:
                                    ret = args3[0];

                                    return true;

                                default:
                                    ret = Add(args3);
                                    return true;
                                }
                            }
                            else {
                                // 乗算の場合

                                // 引数の係数をすべてかけてまとめます。
                                double n = (from t in args2 select t.Value).Aggregate((x, y) => x * y);
                                foreach (Term t in args2) {
                                    t.Value = 1;
                                }

                                if (n == 0) {
                                    // 係数の積が0の場合

                                    // 結果は0
                                    ret = Zero();
                                    return true;
                                }

                                // 定数を除きます。
                                Term[] args3 = (from t in args2 where !(t is Number) select t).ToArray();

                                switch (args3.Length) {
                                case 0:
                                    Debug.Assert(args2.Count == 1 && args2[0] is Number && (args2[0] as Number).Value == 1);

                                    ret = One();
                                    break;

                                case 1:
                                    ret = args3[0];
                                    break;

                                default:
                                    ret = Mul(args3);
                                    break;
                                }

                                (ret as Term).Value = app.Value * n;
                                return true;
                            }
                        }

                        ret = new Apply(app.FunctionApp.VarRef, args1);

                        return true;
                    }

                    return false;
                }) as Term;
        }
    }
}