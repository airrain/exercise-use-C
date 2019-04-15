import java.util.*
abstract class Company{
	protected String name;
	protected Company(String name){
		this.name = name;
	}
	public abstract void add(Company c);
	public abstract void delete(Company c);
}
class ConcreteCompany extends Company{
	private List<Company> children = new ArrayList<Company>();
	public ConcreteCompany(String name){super(name);}
	public void add(Company c){chileren.add(c);}
	public void delete(chileren.remove(c);)	
}
