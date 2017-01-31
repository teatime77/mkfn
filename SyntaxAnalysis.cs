using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace mkfn {
    public partial class mkfn {
        public static mkfn Singleton;

        string HomeDir = @"C:\usr\prj\mkfn\mkfn";
        public string[] Lines;
        List<SimpleType> SimpleTypes = new List<SimpleType>();
        List<ArrayType> ArrayTypes = new List<ArrayType>();
        List<SimpleType> AppClasses = new List<SimpleType>();

        public int TokenPos;
        public Token CurrentToken;
        Token NextToken;
        
        List<Token> TokenList;
        Token EOTToken = new Token(TokenType.EOT, null, -1, -1);

        Token ReadNextToken() {
            Token current_token = CurrentToken;

            CurrentToken = NextToken;

            // トークンの位置を1つ進める。
            TokenPos++;

            if (TokenPos + 1 < TokenList.Count) {

                NextToken = TokenList[TokenPos + 1];
            }
            else{

                NextToken = EOTToken;
            }

            return current_token;
        }

        Token GetToken(string text) {
            if(CurrentToken.Text != text) {

                throw new SyntaxException();
            }

            return ReadNextToken();
        }

        Token GetToken(TokenType type) {
            if (type != TokenType.Any && type != CurrentToken.Type) {

                throw new SyntaxException();
            }

            return ReadNextToken();
        }

        void ReadUsing() {
            GetToken("using");
            GetToken(TokenType.Identifier);

            while (CurrentToken.Text == ".") {

                GetToken(".");
                GetToken(TokenType.Identifier);
            }

            GetToken(";");

        }

        void ReadModifier() {
            while (true) {
                if(CurrentToken.Text == "public") {
                    GetToken(TokenType.Any);
                }
                else if (CurrentToken.Text == "abstract") {
                    GetToken(TokenType.Any);
                }
                else if (CurrentToken.Text == "static") {
                    GetToken(TokenType.Any);
                }
                else if (CurrentToken.Text == "override") {
                    GetToken(TokenType.Any);
                }
                else {
                    break;
                }
            }
        }

        SimpleType GetSimpleType(string name) {

            var v = from c in SimpleTypes where c.Name == name select c;
            if (v.Any()) {

                return v.First();
            }

            SimpleType type = new SimpleType(name);
            SimpleTypes.Add(type);

            return type;
        }

        Class ReadType() {
            Token type_id = GetToken(TokenType.Identifier);

            if (CurrentToken.Text != "[") {
                // 配列でない場合

                return GetSimpleType(type_id.Text);
            }
            else { 
                // 配列の場合


                GetToken("[");

                int dim_cnt = 1;

                while (CurrentToken.Text == ",") {
                    dim_cnt++;
                    GetToken(",");
                }

                GetToken("]");

                var v = from c in ArrayTypes where c.Name == type_id.Text && c.DimCnt == dim_cnt select c;
                if (v.Any()) {

                    return v.First();
                }

                ArrayType type = new ArrayType(GetSimpleType(type_id.Text), dim_cnt);
                ArrayTypes.Add(type);

                return type;
            }
        }

        Variable ReadVariable() {
            Class type = ReadType();
            Token id = GetToken(TokenType.Identifier);

            return new Variable(id.Text, type, null);
        }

        Reference ReadReference() {
            Token id = GetToken(TokenType.Identifier);

            if(CurrentToken.Text != "[") {

                return new Reference(id.Text, null, null);
            }
            else {
                GetToken("[");

                List<Term> idxes = new List<Term>();
                while (true) {
                    Term idx = Expression();
                    idxes.Add(idx);

                    if (CurrentToken.Text == "]") {
                        break;
                    }
                    GetToken(",");
                }
                GetToken("]");

                return new Reference(id.Text, null, idxes.ToArray());
            }
        }

        List<Term> ReadArgs() {
            List<Term> terms = new List<Term>();
            GetToken("(");

            if (CurrentToken.Text != ")") {

                while (true) {
                    terms.Add(Expression());

                    if (CurrentToken.Text == ")") {

                        break;
                    }
                    GetToken(",");
                }
            }
            GetToken(")");

            return terms;

        }

        public Term PrimaryExpression() {
            if(CurrentToken.Type == TokenType.Identifier) {
                Reference r = ReadReference();

                if(CurrentToken.Text != "(") {
                    return r;
                }

                List<Term> args = ReadArgs();
                return new Apply(r, args.ToArray());
            }
            else if (CurrentToken.Type == TokenType.Number) {
                Token num = GetToken(TokenType.Number);

                return new Number(num.Text);
            }
            else if (CurrentToken.Text == "(") {

                GetToken("(");
                Term term = Expression();
                GetToken(")");

                if(CurrentToken.Text == ".") {

                    if (!(term is LINQ)) {

                        throw new SyntaxException();
                    }

                    LINQ lnq = term as LINQ;

                    GetToken(".");
                    Token id = GetToken(TokenType.Identifier);

                    GetToken("(");
                    GetToken(")");

                    if(id.Text == SumFnc.Name) {

                        lnq.Aggregate = new Reference(SumFnc);
                    }
                    else if (id.Text == MaxFnc.Name) {

                        lnq.Aggregate = new Reference(MaxFnc);
                    }
                    else {

                        throw new SyntaxException();
                    }

                    return lnq;
                }

                return term;
            }
            else if (CurrentToken.Text == "from") {
                List<Variable> vars = new List<Variable>();

                while (CurrentToken.Text == "from") {

                    GetToken("from");
                    Token id = GetToken(TokenType.Identifier);
                    GetToken("in");
                    Term domain = Expression();

                    vars.Add(new Variable(id.Text, null, domain));
                }

                GetToken("select");

                Term term = Expression();

                return new LINQ(vars.ToArray(), term, null);
            }
            else if (CurrentToken.Text == "new") {
                List<Term> args = new List<Term>();

                GetToken("new");
                GetToken(TokenType.Identifier);
                GetToken("[");
                while (true) {

                    args.Add( Expression() );
                    if(CurrentToken.Text != ",") {
                        break;
                    }
                    GetToken(",");
                }
                GetToken("]");

                return new Apply(new Reference(NewFnc), args.ToArray());
            }
            else {
                throw new SyntaxException();
            }
        }


        public Term UnaryExpression() {
            if (CurrentToken.Text == "-") {

                GetToken("-");

                Term t1 = PrimaryExpression();

                return new Apply(new Reference(SubFnc), new Term[] { t1 });
            }
            else {

                return PrimaryExpression();
            }
        }

        public Term MultiplicativeExpression() {
            Term t1 = UnaryExpression();

            while (CurrentToken.Text == "*" || CurrentToken.Text == "/") {

                string opr = CurrentToken.Text;

                List<Term> args = new List<Term>();
                args.Add(t1);

                while (CurrentToken.Text == opr) {
                    GetToken(opr);
                    args.Add(UnaryExpression());
                }

                if (opr == "*") {

                    t1 = new Apply(new Reference(MulFnc), args.ToArray());
                }
                else {

                    t1 = new Apply(new Reference(DivFnc), args.ToArray());
                }
            }

            return t1;
        }

        public Term AdditiveExpression() {
            Term t1 = MultiplicativeExpression();

            while (CurrentToken.Text == "+" || CurrentToken.Text == "-") {

                string opr = CurrentToken.Text;

                List<Term> args = new List<Term>();
                args.Add(t1);


                while(CurrentToken.Text == opr) {
                    GetToken(opr);
                    args.Add( MultiplicativeExpression() );
                }

                if(opr == "+") {

                    t1 = new Apply(new Reference(AddFnc), args.ToArray());
                }
                else {

                    t1 = new Apply(new Reference(SubFnc), args.ToArray());
                }
            }

            return t1;
        }


        Term Expression() {
            return AdditiveExpression();
        }

        List<Statement> ReadBlockStatement() {
            List<Statement> statements = new List<Statement>();
            GetToken("{");

            while(CurrentToken.Text != "}") {
                statements.Add( ReadStatement() );
            }
            GetToken("}");

            return statements;
        }

        Statement ReadStatement() {
            if(CurrentToken.Type == TokenType.Identifier) {
                Reference left = ReadReference();

                GetToken("=");

                Term right = Expression();
                GetToken(";");

                Assignment asn = new Assignment(left, right);
                Debug.WriteLine(asn.ToString());
                return asn;
            }
            else if(CurrentToken.Text == "return") {

                GetToken("return");
                Term value = Expression();
                GetToken(";");

                return new Return(value);
            }
            else if(CurrentToken.Text == "foreach") {

                GetToken("foreach");
                GetToken("(");
                Variable v = ReadVariable();
                GetToken("in");
                v.Domain = Expression();
                GetToken(")");

                List<Statement> stmts = ReadBlockStatement();

                return new ForEach(v, stmts);
            }
            else {
                throw new SyntaxException();
            }
        }

        Function ReadFunction(string name, Class type) {
            Function fnc = new Function(name, type);

            GetToken("(");

            if (CurrentToken.Text != ")") {

                while (true) {
                    Variable arg = ReadVariable();

                    fnc.Params.Add(arg);

                    if (CurrentToken.Text == ")") {

                        break;
                    }
                    GetToken(",");
                }
            }
            GetToken(")");

            if (CurrentToken.Text == ";") {

                GetToken(";");
            }
            else {

                fnc.Statement = new BlockStatement(ReadBlockStatement());
            }

            return fnc;
        }

        void ReadClass() {
            ReadModifier();

            GetToken("class");
            Token class_id = GetToken(TokenType.Identifier);
            SimpleType cls = GetSimpleType(class_id.Text);
            
            if (CurrentToken.Text == ":") {

                GetToken(":");
                Token super_class = GetToken(TokenType.Identifier);

                cls.Parent = GetSimpleType(super_class.Text);
            }
            GetToken("{");

            while (CurrentToken.Text != "}") {

                ReadModifier();
                Class type = ReadType();

                if (CurrentToken.Text == "(") {
                    // コンストラクターの場合

                    cls.Functions.Add(ReadFunction("new", type));
                }
                else {
                    // コンストラクターでない場合

                    Token id = GetToken(TokenType.Identifier);

                    if (CurrentToken.Text == ";") {
                        // フィールドの場合

                        GetToken(";");

                        Variable field = new Variable(id.Text, type, null);

                        cls.Fields.Add(field);

                        Debug.WriteLine("field {0}", field.ToString(), "");
                    }
                    else {
                        // メソッドの場合

                        cls.Functions.Add(ReadFunction(id.Text, type));
                    }
                }
            }

            GetToken("}");
        }

        void ParseSource() {
            TokenPos = 0;
            CurrentToken = TokenList[0];
            NextToken = TokenList[1];

            while (CurrentToken.Text == "using") {

                ReadUsing();
            }

            while (CurrentToken != EOTToken) {

                ReadClass();
            }
        }

        public void ParseSourceFile(string path) {
            Lines = File.ReadAllLines(path, Encoding.UTF8);

            TokenList = new List<Token>();
            for (int i = 0; i < Lines.Length; i++) {
                TokenList.AddRange(LexicalAnalysis(Lines[i], i));
            }

            ParseSource();
        }

        public void Main() {
            ParseSourceFile(HomeDir + @"\NeuralNetwork.cs");
        }
    }


    public class Class {
        public Class Parent;
        public string Name;

        public bool IsSubClass(Class cls) {
            return Parent == cls;
        }
    }

    public class SimpleType : Class {
        public List<Variable> Fields = new List<Variable>();
        public List<Function> Functions = new List<Function>();

        public SimpleType(string name) {
            Name = name;
        }

        public override string ToString() {
            return Name;
        }
    }

    public class ArrayType : Class {
        public Class ElementType;
        public int DimCnt;

        public ArrayType(Class element_type, int dim_cnt) {
            ElementType = element_type;
            DimCnt = dim_cnt;
        }

        public override string ToString() {
            return Name + "[" + new string(',', DimCnt - 1) + "]";
        }
    }

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
            if(TypeVar == null) {
                return Name;
            }
            return TypeVar.ToString() + " " + Name;
        }
    }

    public class Function : Variable {
        public List<Variable> Params = new List<Variable>();

        public BlockStatement Statement;
        public Function(string name, Class type) : base(name, type, null) {
        }
    }

    public class Statement {
        public object ParentStmt;
    }

    public class Assignment : Statement {
        public Term Left;
        public Term Right;

        public Assignment(Term left, Term right) {
            Left = left;
            Right = right;

            Left.Parent = this;
            Right.Parent = this;
        }

        public override string ToString() {
            return Left.ToString() + " = " + Right.ToString();
        }
    }

    public class Return : Statement {
        public Term Value;

        public Return(Term value) {
            Value = value;
        }
    }

    public class BlockStatement : Statement {
        public List<Statement> Statements;

        public BlockStatement(List<Statement> statements) {
            Statements = statements;
        }
    }

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

    public abstract class Term {
        public object Parent;

        public virtual bool Eq(Object obj) {
            return false;
        }

        public Term Clone(Dictionary<Variable, Variable> var_tbl) {
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

    public class Number : Term {
        public double Value;

        public Number(string text) {

            if(! double.TryParse(text, out Value)) {
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

        public override string ToString() {
            if (Indexes == null) {
                return Name;
            }
            else {
                return Name + "[" + string.Join(", ", from x in Indexes select x.ToString()) + "]";
            }
        }
    }

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

    public class SyntaxException : Exception {

        public SyntaxException() {
            Token tkn = mkfn.Singleton.CurrentToken;
            string line = mkfn.Singleton.Lines[tkn.LineIndex];
            Debug.WriteLine("Err {0}行: {1} ^ {2}", tkn.LineIndex + 1, line.Substring(0, tkn.CharPos), line.Substring(tkn.CharPos));
            Debug.Write("");
        }
    }
}