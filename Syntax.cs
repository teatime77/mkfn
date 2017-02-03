using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace mkfn {
    /*
        クラス
    */
    public class Class {
        public Class Parent;
        public string Name;
        public List<Variable> Fields = new List<Variable>();
        public List<Function> Functions = new List<Function>();

        public Class(string name) {
            Name = name;
        }

        public virtual Class ValueType() {
            return this;
        }

        public override string ToString() {
            return Name;
        }

        public bool IsSubClass(Class cls) {
            return Parent == cls;
        }
    }

    /*
        配列の型
    */
    public class ArrayType : Class {
        public Class ElementType;
        public int DimCnt;

        public ArrayType(Class element_type, int dim_cnt) : base(element_type.Name) {
            ElementType = element_type;
            DimCnt = dim_cnt;
        }

        public override Class ValueType() {
            return ElementType;
        }

        public override string ToString() {
            return Name + "[" + new string(',', DimCnt - 1) + "]";
        }
    }

    /*
        変数
    */
    public class Variable {
        public object ParentVar;
        public string Name;
        public Class TypeVar;
        public Term Domain;

        public Variable(string name, Class type, Term domain) {
            Name = name;
            TypeVar = type;
            Domain = domain;

            if (Domain != null) {
                Domain.Parent = this;
            }
        }

        public Variable Clone(Dictionary<Variable, Variable> var_tbl) {
            Term domain = (Domain == null ? null : Domain.Clone(var_tbl));
            Variable v1 = new Variable(Name, TypeVar, domain);
            var_tbl.Add(this, v1);

            return v1;
        }

        public override string ToString() {
            if (TypeVar == null) {
                return Name;
            }
            return TypeVar.ToString() + " " + Name;
        }
    }

    /*
        関数
    */
    public class Function : Variable {
        public List<Variable> Params = new List<Variable>();

        public BlockStatement Statement;
        public Function(string name, Class type) : base(name, type, null) {
        }
    }

    /*
        文
    */
    public class Statement {
        public object ParentStmt;
    }

    /*
        代入文
    */
    public class Assignment : Statement {
        public Reference Left;
        public Term Right;

        public Assignment(Reference left, Term right) {
            Left = left;
            Right = right;

            Left.Parent = this;
            Right.Parent = this;
        }

        public override string ToString() {
            return Left.ToString() + " = " + Right.ToString();
        }
    }

    /*
        return文
    */
    public class Return : Statement {
        public Term Value;

        public Return(Term value) {
            Value = value;
        }
    }

    /*
        ブロック文
    */
    public class BlockStatement : Statement {
        public List<Statement> Statements;

        public BlockStatement(List<Statement> statements) {
            Statements = statements;

            foreach (Statement s in Statements) {
                s.ParentStmt = this;
            }
        }
    }

    /*
        foreach文
    */
    public class ForEach : BlockStatement {
        public Variable LoopVariable;

        public ForEach(Variable variable, List<Statement> statements) : base(statements) {
            LoopVariable = variable;

            LoopVariable.ParentVar = this;
            foreach (Statement s in Statements) {
                s.ParentStmt = this;
            }
        }
    }

    /*
        項
    */
    public abstract class Term {
        public object Parent;

        public virtual bool Eq(Object obj) {
            return false;
        }

        public Term Clone(Dictionary<Variable, Variable> var_tbl = null) {
            if (var_tbl == null) {
                var_tbl = new Dictionary<Variable, Variable>();
            }

            if (this is Reference) {
                return (this as Reference).Clone(var_tbl);
            }
            else if (this is Number) {
                return (this as Number).Clone(var_tbl);
            }
            else if (this is Apply) {
                return (this as Apply).Clone(var_tbl);
            }
            else if (this is LINQ) {
                return (this as LINQ).Clone(var_tbl);
            }
            else {
                Debug.Assert(false);
                return null;
            }
        }
    }

    /*
        数値定数
    */
    public class Number : Term {
        public double Value;

        public Number(string text) {

            if (!double.TryParse(text, out Value)) {
                throw new SyntaxException();
            }
        }

        public Number(double d) {
            Value = d;
        }

        public new Number Clone(Dictionary<Variable, Variable> var_tbl) {
            return new Number(Value);
        }

        public override bool Eq(Object obj) {
            if (!(obj is Number)) {
                return false;
            }
            return Value == (obj as Number).Value;
        }

        public override string ToString() {
            return Value.ToString();
        }
    }

    /*
        変数参照
    */
    public class Reference : Term {
        public string Name;
        public Variable VarRef;
        public Term[] Indexes;

        public Reference(string name, Variable ref_var, Term[] idx) {
            Name = name;
            VarRef = ref_var;
            Indexes = idx;

            if (Indexes != null) {
                foreach (Term t in Indexes) {
                    t.Parent = this;
                }
            }
        }

        public Reference(Variable v) {
            Name = v.Name;
            VarRef = v;
            Indexes = null;
        }

        public new Reference Clone(Dictionary<Variable, Variable> var_tbl) {
            if (var_tbl == null) {
                var_tbl = new Dictionary<Variable, Variable>();
            }
            Variable v1;
            if (!var_tbl.TryGetValue(VarRef, out v1)) {
                v1 = VarRef;
            }

            if (Indexes == null) {
                return new Reference(Name, v1, null);
            }

            Term[] idx = (from t in Indexes select t.Clone(var_tbl)).ToArray();

            return new Reference(Name, v1, idx);
        }

        public override bool Eq(Object obj) {
            if (!(obj is Reference)) {
                return false;
            }
            Reference r = obj as Reference;
            if (r.VarRef != VarRef) {
                return false;
            }
            if ((Indexes == null) != (r.Indexes == null)) {
                return false;
            }
            if (Indexes == null) {
                return true;
            }
            else {
                Debug.Assert(Indexes.Length == r.Indexes.Length);
                for (int i = 0; i < Indexes.Length; i++) {
                    if (!Indexes[i].Eq(r.Indexes[i])) {
                        return false;
                    }
                }
                return true;
            }
        }

        public bool Defined() {
            return Parent is Assignment && (Parent as Assignment).Left == this;
        }

        public bool Used() {
            return !Defined();
        }

        public override string ToString() {
            if (Indexes == null) {
                return Name;
            }
            else {
                return Name + "[" + string.Join(", ", from x in Indexes select x.ToString()) + "]";
            }
        }
    }

    /*
        関数適用
    */
    public class Apply : Term {
        public Reference Function;
        public Term[] Args;

        public Apply(Reference function, Term[] args) {
            Function = function;
            Args = args;

            Function.Parent = this;
            foreach (Term t in Args) {
                t.Parent = this;
            }
        }

        public Apply(Variable function, params Term[] args) : this(new Reference(function), args) {
        }

        public new Apply Clone(Dictionary<Variable, Variable> var_tbl) {
            Term[] args = (from t in Args select t.Clone(var_tbl)).ToArray();
            return new Apply(Function.Clone(var_tbl), args);
        }

        public override string ToString() {
            if ("+-*/%".Contains(Function.Name[0])) {
                string s;

                if (Args.Length == 1) {

                    s = Function.Name + " " + Args[0].ToString();
                }
                else {

                    s = string.Join(" " + Function.Name + " ", from x in Args select x.ToString());
                }

                if (Parent is Apply && (Parent as Apply).Precedence() <= Precedence()) {
                    return "(" + s + ")";
                }
                else {
                    return s;
                }
            }
            else {

                return Function.Name + "(" + string.Join(", ", from x in Args select x.ToString()) + ")";
            }
        }

        public override bool Eq(Object obj) {
            if (!(obj is Apply)) {
                return false;
            }

            Apply app = obj as Apply;

            if ( ! Function.Eq(app.Function) ) {
                return false;
            }
            if (Args.Length != app.Args.Length) {
                return false;
            }

            for (int i = 0; i < Args.Length; i++) {
                if (!Args[i].Eq(app.Args[i])) {
                    return false;
                }
            }
            return true;
        }

        public int Precedence() {
            if (Char.IsLetter(Function.Name[0])) {
                return 20;
            }

            if ("*/%".Contains(Function.Name[0])) {
                return 1;
            }

            if (Args.Length == 1 && "+-!".Contains(Function.Name[0])) {
                return 2;
            }

            return 10;
        }
    }

    /*
        LINQ
    */
    public class LINQ : Term {
        public Variable[] Variables;
        public Term Select;
        public Reference Aggregate;

        public LINQ(Variable[] variables, Term select, Reference aggregate) {
            Variables = variables;
            Select = select;
            Aggregate = aggregate;

            foreach (Variable v in Variables) {
                v.ParentVar = this;
            }
            Select.Parent = this;
            if (Aggregate != null) {

                Aggregate.Parent = this;
            }
        }

        public new LINQ Clone(Dictionary<Variable, Variable> var_tbl) {
            Variable[] vars = (from v in Variables select v.Clone(var_tbl)).ToArray();
            return new LINQ(vars, Select.Clone(var_tbl), (Aggregate == null ? null : Aggregate.Clone(var_tbl)));
        }

        public override string ToString() {
            string list = string.Join(" ", from x in Variables select "from " + x.Name + " in " + x.Domain.ToString()) + " select " + Select.ToString();
            if (Aggregate == null) {

                return list;
            }
            return "(" + list + ")." + Aggregate.Name + "()";
        }
    }

    /*
        構文エラー
    */
    public class SyntaxException : Exception {

        public SyntaxException() {
            Token tkn = mkfn.Singleton.CurrentToken;
            string line = mkfn.Singleton.Lines[tkn.LineIndex];
            Debug.WriteLine("Err {0}行: {1} ^ {2}", tkn.LineIndex + 1, line.Substring(0, tkn.CharPos), line.Substring(tkn.CharPos));
            Debug.Write("");
        }
    }
}
