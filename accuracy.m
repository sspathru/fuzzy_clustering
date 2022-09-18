function [acc] = accuracy(U, GTL)

Ucrisp = zeros(size(U));
for i = 1:size(U,1)
   [~,a] = (max(U(i,:)));
    Ucrisp(i,a) = 1;
end

for i = 1:size(Ucrisp,2)
    Ucrisp2(:,i) = Ucrisp(:,i) .* GTL(:,1);
end

for i = 1:size(Ucrisp,2)
    ff = nonzeros(Ucrisp2(:,i));
    cnames(1,i) = mode(ff);
end

Ucrisp3 = cnames .* Ucrisp;
Ucrisp4 = max(Ucrisp3,[],2);
true_labels = sum(Ucrisp4 == GTL);
acc = true_labels/max(size(GTL));

end

